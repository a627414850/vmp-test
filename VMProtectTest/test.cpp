#include "pch.h"

#include "AbstractStream.hpp"
#include "CFG.hpp"

// global, i mean... why can't triton have userdefined-context for callback functions?
static AbstractStream* g_stream = nullptr;
static triton::uint64 g_module_base = 0x00400000;
static triton::uint64 g_vmp0_address = 0x17000;
static triton::uint64 g_vmp0_size = 0x86CB0;
static triton::uint64 g_vmp_section_address = (g_module_base + g_vmp0_address);
static triton::uint64 g_vmp_section_size = g_vmp0_size;
static triton::uint64 g_vmp_stackpointer, g_x86_stackpointer;

// handler
static triton::engines::symbolic::SharedSymbolicVariable g_symvar_bp, g_symvar_sp;

struct VMP_CPU_STATE
{
	triton::uint64 rbx, rsi, rdi, rbp, rsp;
	bool x64;
};

// triton::ast::search but std::set instead
static std::set<triton::ast::SharedAbstractNode> _collect_variable_nodes(const triton::ast::SharedAbstractNode& node)
{
	std::set<triton::ast::SharedAbstractNode> result;
	if (!node)
		return result;

	std::stack<triton::ast::AbstractNode*>                worklist;
	std::unordered_set<const triton::ast::AbstractNode*>  visited;

	worklist.push(node.get());
	while (!worklist.empty()) {
		auto current = worklist.top();
		worklist.pop();

		// This means that node is already in work_stack and we will not need to convert it second time
		if (visited.find(current) != visited.end()) {
			continue;
		}

		visited.insert(current);
		if (current->getType() == triton::ast::VARIABLE_NODE)
			result.insert(current->shared_from_this());

		if (current->getType() == triton::ast::REFERENCE_NODE) {
			worklist.push(reinterpret_cast<triton::ast::ReferenceNode*>(current)->getSymbolicExpression()->getAst().get());
		}
		else {
			for (const auto& child : current->getChildren()) {
				worklist.push(child.get());
			}
		}
	}
	return result;
}

// callbacks
static void vm_enter_callback(triton::API& ctx, const triton::arch::MemoryAccess& mem)
{
	if (!ctx.isConcreteMemoryValueDefined(mem) &&
		(g_vmp_section_address <= mem.getAddress() && mem.getAddress() < (g_vmp_section_address + g_vmp_section_size)))
	{
		triton::uint8 buf[32];
		g_stream->seek(mem.getAddress());
		if (g_stream->read(buf, mem.getSize()) != mem.getSize())
			throw std::runtime_error("stream.read failed");
		
		ctx.setConcreteMemoryAreaValue(mem.getAddress(), buf, mem.getSize());
		ctx.taintMemory(mem);
	}
}
static void vm_handler_callback(triton::API& ctx, const triton::arch::MemoryAccess& mem)
{
	const triton::uint64 runtime_address = mem.getAddress();
	if (!ctx.isConcreteMemoryValueDefined(mem) &&
		(g_vmp_section_address <= runtime_address && runtime_address < (g_vmp_section_address + g_vmp_section_size)))
	{
		triton::uint8 buf[32];
		g_stream->seek(runtime_address);
		if (g_stream->read(buf, mem.getSize()) != mem.getSize())
			throw std::runtime_error("stream.read failed");

		ctx.taintMemory(mem);
		ctx.setConcreteMemoryAreaValue(runtime_address, buf, mem.getSize());
	}
	else
	{
		auto leaAst = mem.getLeaAst();
		if (leaAst && leaAst->isSymbolized())
		{
			auto symvar = ctx.symbolizeMemory(mem);
			if (g_vmp_stackpointer <= runtime_address && runtime_address < (g_vmp_stackpointer + 0x10))
			{
				// load from stack, ast: ebp+offset
				char buf[256];
				sprintf_s(buf, 256, "arg_%llx_%d", runtime_address - g_vmp_stackpointer, mem.getSize());
				symvar->setAlias(buf);
			}
			else if (g_x86_stackpointer <= runtime_address && runtime_address < g_vmp_stackpointer)
			{
				// load from scratch, ast: esp+offset
				char buf[256];
				sprintf_s(buf, 256, "scratch_%llx_%d", runtime_address - g_x86_stackpointer, mem.getSize());
				symvar->setAlias(buf);
			}
		}
	}
}


static void _runtime_optimize(std::shared_ptr<triton::API> triton_api, AbstractStream& stream,
	triton::uint64& runtime_address, VMP_CPU_STATE& cpu_state, std::list<std::shared_ptr<x86_instruction>>& instructions_o)
{
	std::list<std::shared_ptr<x86_instruction>> instructions;
	std::map<std::shared_ptr<x86_instruction>, triton::uint64> replaceable;
	std::map<std::shared_ptr<x86_instruction>, triton::arch::MemoryAccess> memory_replaceable;
	for (;;)
	{
		stream.seek(runtime_address);
		const std::shared_ptr<x86_instruction> xed_instruction = stream.readNext();
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();

		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());

		// disam
		triton_api->disassembly(triton_instruction);
		for (auto& op : triton_instruction.operands)
		{
			if (op.getType() == triton::arch::OP_MEM)
			{
				const triton::arch::MemoryAccess& mem = op.getConstMemory();
				const triton::arch::Register& base_reg = mem.getConstBaseRegister();
				const triton::arch::Register& index_reg = mem.getConstIndexRegister();
				const triton::arch::Immediate& displacement = mem.getConstDisplacement();

				triton::arch::MemoryAccess new_mem;
				bool updated = false;

				triton::uint64 value = displacement.getValue();
				if (triton_api->isRegisterValid(base_reg) && !triton_api->isRegisterSymbolized(base_reg))
				{
					value += triton_api->getConcreteRegisterValue(base_reg).convert_to<triton::uint64>();
					updated = true;
				}
				else
					new_mem.setBaseRegister(base_reg);

				if (triton_api->isRegisterValid(index_reg) && !triton_api->isRegisterSymbolized(index_reg))
				{
					value += triton_api->getConcreteRegisterValue(index_reg).convert_to<triton::uint64>() * mem.getConstScale().getValue();
					updated = true;
				}
				else
				{
					new_mem.setIndexRegister(index_reg);
					new_mem.setScale(mem.getConstScale());
				}

				triton::arch::Immediate new_displacement(value, 4);
				if (updated && value == new_displacement.getValue())
				{
					new_mem.setDisplacement(new_displacement);
					memory_replaceable.insert(std::make_pair(xed_instruction, new_mem));
				}
			}
		}

		// processing
		if (!triton_api->processing(triton_instruction))
		{
			throw std::runtime_error("triton processing failed");
		}

		const auto& op0 = triton_instruction.operands[0];
		const triton::arch::Register& op0_reg = op0.getConstRegister();
		const triton::arch::MemoryAccess& op0_mem = op0.getConstMemory();
		switch (triton_instruction.getType())
		{
			// unary
			case triton::arch::x86::ID_INS_NEG:
			case triton::arch::x86::ID_INS_NOT:
			case triton::arch::x86::ID_INS_BSWAP:

				// binary
			case triton::arch::x86::ID_INS_MOV:
			case triton::arch::x86::ID_INS_MOVZX:
			case triton::arch::x86::ID_INS_MOVSX:
			case triton::arch::x86::ID_INS_ADD:
			case triton::arch::x86::ID_INS_SUB:
			case triton::arch::x86::ID_INS_XOR:
			case triton::arch::x86::ID_INS_AND:
			case triton::arch::x86::ID_INS_OR:
			case triton::arch::x86::ID_INS_ROL:
			case triton::arch::x86::ID_INS_ROR:

			case triton::arch::x86::ID_INS_LEA:
			{
				if (triton_api->isRegisterValid(op0_reg) && !triton_api->isRegisterSymbolized(op0_reg))
				{
					replaceable.insert(std::make_pair(xed_instruction,
						triton_api->getConcreteRegisterValue(op0_reg).convert_to<triton::uint64>()));
				}
				else if (op0_mem.getLeaAst() && !triton_api->isMemorySymbolized(op0_mem))
				{
					replaceable.insert(std::make_pair(xed_instruction,
						triton_api->getConcreteMemoryValue(op0_mem).convert_to<triton::uint64>()));
				}
				break;
			}

			// etc
			case triton::arch::x86::ID_INS_PUSH:
			{
				triton::uint32 mem_size;
				if (op0.getBitSize() == 64) mem_size = 8;
				else if (op0.getBitSize() == 32) mem_size = 4;
				else if (op0.getBitSize() == 16) mem_size = 2;
				else throw std::runtime_error("");

				const triton::arch::Register& sp = triton_api->getCpuInstance()->getStackPointer();
				triton::arch::MemoryAccess mem(triton_api->getConcreteRegisterValue(sp).convert_to<triton::uint64>(), mem_size);
				if (!triton_api->isMemorySymbolized(mem))
				{
					replaceable.insert(std::make_pair(xed_instruction,
						triton_api->getConcreteMemoryValue(mem).convert_to<triton::uint64>()));
				}
				break;
			}
			default:
				break;
		}


		if (!triton_instruction.isBranch())
		{
			//std::cout << triton_instruction << "\n";
			instructions.push_back(xed_instruction);
		}


		const triton::arch::Register& ip = triton_api->getCpuInstance()->getProgramCounter();
		runtime_address = triton_api->getConcreteRegisterValue(ip).convert_to<triton::uint64>();
		if (triton_api->isRegisterSymbolized(ip))
		{
			// if ip = AST_OF(sp, bp), it is probably stack check
			bool _exit = false;
			auto ip_ast = triton_api->getRegisterAst(ip);
			for (const auto& node : _collect_variable_nodes(ip_ast))
			{
				const auto& symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
				if (symvar->getId() != g_symvar_sp->getId() && symvar->getId() != g_symvar_bp->getId())
				{
					// probably exit
					_exit = true;
					std::cout << "ip="
						<< triton_api->processSimplification(ip_ast, true)
						<< "\n";
					break;
				}
			}

			/*
			if (!_exit && !triton_instruction.isConditionTaken())
			{
				auto branches = triton_api->getPathConstraints().at(0).getBranchConstraints();
				auto br1 = branches.at(0);
				if ((std::get<0>(br1) != false) ||
					(std::get<1>(br1) != 0x123) ||
					(std::get<2>(br1) != 0x100) ||
					(std::get<3>(br1)->getHash() != node->getHash())) {
					std::cerr << "test_9: KO (wrong br1)" << std::endl;
					return 1;
				}
			}*/

			if (_exit)
				break;
		}
		else if (triton_api->isRegisterTainted(ip))
		{
			// ip = ast(pcode)
			break;
		}
		else if (triton_instruction.getType() == triton::arch::x86::ID_INS_RET)
		{
			// if sp = ast(bp) -> vm exit
			auto sp_ast = triton_api->getRegisterAst(triton_api->getCpuInstance()->getStackPointer());
			if (auto d = _collect_variable_nodes(sp_ast);
				d.size() == 1
				&& std::dynamic_pointer_cast<triton::ast::VariableNode>(*d.begin())->getSymbolicVariable()->getId() == g_symvar_bp->getId())
			{
				std::cout << "sp="
					<< triton_api->processSimplification(sp_ast, true)
					<< "\n";

				break;
			}
		}
	}

	// uh
	bool is_control_flow = false;
	const triton::arch::Register& bytecode_reg = cpu_state.x64 ? triton_api->registers.x86_rsi : triton_api->registers.x86_esi;
	if (triton_api->isRegisterSymbolized(bytecode_reg))
	{
		std::cout << "bytecode="
			<< triton_api->processSimplification(triton_api->getRegisterAst(bytecode_reg), 1)
			<< "\n";
		is_control_flow = true;
		runtime_address = 0;
	}

	// replace
	for (auto xed_instruction : instructions)
	{
		if (auto it = memory_replaceable.find(xed_instruction); it != memory_replaceable.end())
		{
			//std::cout << "memory_replaceable: " << xed_instruction->get_string() << "\n";
			const triton::arch::MemoryAccess& mem = it->second;
			if (xed_instruction->get_number_of_memory_operands() != 1)
				throw std::runtime_error("there's too many or less memory operands");

			xed_instruction->encoder_init_from_decode();
			xed_instruction->encoder_set_memory_displacement(mem.getConstDisplacement().getValue(), mem.getConstDisplacement().getSize());
			if (!triton_api->isRegisterValid(mem.getConstBaseRegister()))
			{
				// remove
				xed_instruction->encoder_set_base0(XED_REG_INVALID);
			}
			if (!triton_api->isRegisterValid(mem.getConstIndexRegister()))
			{
				// remove
				xed_instruction->encoder_set_index(XED_REG_INVALID);
				//xed_instruction->encoder_set_scale(0);
			}
			xed_instruction->encode();
		}

		if (auto it = replaceable.find(xed_instruction); it != replaceable.end())
		{
			//std::cout << "replaceable: " << xed_instruction->get_string() << "\n";
			switch (xed_instruction->get_iclass())
			{
				// unary
				case XED_ICLASS_NEG:
				case XED_ICLASS_NOT:
				case XED_ICLASS_BSWAP:

					// binary
				case XED_ICLASS_MOV:
				case XED_ICLASS_MOVSX:
				case XED_ICLASS_MOVZX:
				case XED_ICLASS_ADD:
				case XED_ICLASS_SUB:
				case XED_ICLASS_XOR:
				case XED_ICLASS_AND:
				case XED_ICLASS_OR:
				case XED_ICLASS_ROL:
				case XED_ICLASS_ROR:
				{
					if (!xed_instruction->get_operand(0).is_memory() || xed_instruction->get_operand_length_bits(0) <= 32)
					{
						xed_instruction->encoder_init_from_decode();
						xed_instruction->encoder_set_iclass(XED_ICLASS_MOV);
						xed_instruction->encoder_set_operand_order(1, XED_OPERAND_IMM0);
						xed_instruction->encoder_set_uimm0_bits(it->second, xed_instruction->get_operand_length_bits(0));
						xed_instruction->encode();
					}
					break;
				}
				case XED_ICLASS_LEA:
				{
					xed_instruction->encoder_init_from_decode();
					xed_instruction->encoder_set_iclass(XED_ICLASS_MOV);
					xed_instruction->encoder_set_operand_order(1, XED_OPERAND_IMM0);
					xed_instruction->encoder_set_uimm0_bits(it->second, xed_instruction->get_operand_length_bits(0));
					xed_instruction->encode();
					break;
				}
				case XED_ICLASS_PUSH:
				{
					// there's no push imm64 :|
					if (xed_instruction->get_operand_length_bits(0) <= 32)
					{
						xed_instruction->encoder_init_from_decode();
						xed_instruction->encoder_set_iclass(XED_ICLASS_PUSH);
						xed_instruction->encoder_set_operand_order(0, XED_OPERAND_IMM0);
						xed_instruction->encoder_set_uimm0_bits(it->second, xed_instruction->get_operand_length_bits(0));
						xed_instruction->encode();
					}
					break;
				}
				default:
				{
					std::cout << xed_instruction->get_string() << "dd\n";
					throw std::invalid_argument("f");
					break;
				}
			}
		}
	}

	// deob
	std::map<x86_register, bool> dead_registers;
	std::vector<x86_register> dead_ =
	{
		XED_REG_RAX, XED_REG_RCX, XED_REG_RDX,
		//XED_REG_RBX, XED_REG_RSI, XED_REG_RDI,

		XED_REG_R8, XED_REG_R9, XED_REG_R10, XED_REG_R11, 
		XED_REG_R12, XED_REG_R13, XED_REG_R14, XED_REG_R15
	};
	if (!is_control_flow)
	{
		for (int i = 0; i < dead_.size(); i++)
		{
			const x86_register& reg = dead_[i];
			dead_registers[reg.get_gpr8_low()] = true;
			dead_registers[reg.get_gpr8_high()] = true;
			dead_registers[reg.get_gpr16()] = true;
			dead_registers[reg.get_gpr32()] = true;
			dead_registers[reg] = true;
		}
	}
	xed_uint32_t dead_flags = 0xFFFFFFFF;
	apply_dead_store_elimination(instructions, dead_registers, dead_flags);

	// remove push-ret
	for (auto it = instructions.begin(); it != instructions.end();)
	{
		auto push = *it;
		auto ret_it = std::next(it);
		if (ret_it == instructions.end())
			break;

		auto ret = *ret_it;
		if (push->get_iclass() == XED_ICLASS_PUSH
			&& ret->get_iclass() == XED_ICLASS_RET_NEAR)
		{
			instructions.erase(it);
			it = instructions.erase(ret_it);
		}
		else
			++it;
	}

	if (cpu_state.x64)
	{
		cpu_state.rbx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rbx).convert_to <triton::uint64>();
		cpu_state.rsi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rsi).convert_to <triton::uint64>();
		cpu_state.rdi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rdi).convert_to <triton::uint64>();
		cpu_state.rbp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rbp).convert_to <triton::uint64>();
		cpu_state.rsp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_rsp).convert_to <triton::uint64>();
	}
	else
	{
		cpu_state.rbx = triton_api->getConcreteRegisterValue(triton_api->registers.x86_ebx).convert_to <triton::uint64>();
		cpu_state.rsi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_esi).convert_to <triton::uint64>();
		cpu_state.rdi = triton_api->getConcreteRegisterValue(triton_api->registers.x86_edi).convert_to <triton::uint64>();
		cpu_state.rbp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_ebp).convert_to <triton::uint64>();
		cpu_state.rsp = triton_api->getConcreteRegisterValue(triton_api->registers.x86_esp).convert_to <triton::uint64>();
	}
	instructions_o.insert(instructions_o.end(), instructions.begin(), instructions.end());
}

static void _runtime_optimize_enter(AbstractStream& stream,
	triton::uint64& runtime_address, VMP_CPU_STATE& cpu_state, std::list<std::shared_ptr<x86_instruction>>& instructions_o)
{
	auto triton_api = std::make_shared<triton::API>();
	triton_api->setArchitecture(cpu_state.x64 ? triton::arch::ARCH_X86_64 : triton::arch::ARCH_X86);
	triton_api->setAstRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);
	triton_api->addCallback(vm_enter_callback);

	triton_api->setMode(triton::modes::TAINT_THROUGH_POINTERS, true);
	triton_api->enableTaintEngine(true);

	if (cpu_state.x64)
	{
		triton_api->symbolizeRegister(triton_api->registers.x86_rax);
		triton_api->symbolizeRegister(triton_api->registers.x86_rbx);
		triton_api->symbolizeRegister(triton_api->registers.x86_rcx);
		triton_api->symbolizeRegister(triton_api->registers.x86_rdx);
		triton_api->symbolizeRegister(triton_api->registers.x86_rdi);
		triton_api->symbolizeRegister(triton_api->registers.x86_rsi);
		triton_api->symbolizeRegister(triton_api->registers.x86_rbp);
		triton_api->symbolizeRegister(triton_api->registers.x86_eflags);
		triton_api->symbolizeRegister(triton_api->registers.x86_rsp);

		triton_api->symbolizeRegister(triton_api->registers.x86_r8);
		triton_api->symbolizeRegister(triton_api->registers.x86_r9);
		triton_api->symbolizeRegister(triton_api->registers.x86_r10);
		triton_api->symbolizeRegister(triton_api->registers.x86_r11);
		triton_api->symbolizeRegister(triton_api->registers.x86_r12);
		triton_api->symbolizeRegister(triton_api->registers.x86_r13);
		triton_api->symbolizeRegister(triton_api->registers.x86_r14);
		triton_api->symbolizeRegister(triton_api->registers.x86_r15);
	}
	else
	{
		triton_api->symbolizeRegister(triton_api->registers.x86_eax);
		triton_api->symbolizeRegister(triton_api->registers.x86_ebx);
		triton_api->symbolizeRegister(triton_api->registers.x86_ecx);
		triton_api->symbolizeRegister(triton_api->registers.x86_edx);
		triton_api->symbolizeRegister(triton_api->registers.x86_edi);
		triton_api->symbolizeRegister(triton_api->registers.x86_esi);
		triton_api->symbolizeRegister(triton_api->registers.x86_ebp);
		triton_api->symbolizeRegister(triton_api->registers.x86_eflags);
		triton_api->symbolizeRegister(triton_api->registers.x86_esp);
	}
	_runtime_optimize(triton_api, stream, runtime_address, cpu_state, instructions_o);
}
static void _runtime_optimize_handler(AbstractStream& stream,
	triton::uint64& runtime_address, VMP_CPU_STATE& cpu_state, std::list<std::shared_ptr<x86_instruction>>& instructions_o)
{
	auto triton_api = std::make_shared<triton::API>();
	triton_api->setArchitecture(cpu_state.x64 ? triton::arch::ARCH_X86_64 : triton::arch::ARCH_X86);
	triton_api->setAstRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);
	triton_api->addCallback(vm_handler_callback);

	triton_api->setMode(triton::modes::TAINT_THROUGH_POINTERS, true);
	triton_api->enableTaintEngine(true);
	if (cpu_state.x64)
	{
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rbx, cpu_state.rbx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rsi, cpu_state.rsi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rdi, cpu_state.rdi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rbp, cpu_state.rbp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_rsp, cpu_state.rsp);

		triton::uint64 arg0 = cpu_state.rbp;
		triton_api->setConcreteMemoryAreaValue(cpu_state.rbp, (const triton::uint8*) &arg0, 8);

		//triton_api->taintRegister(triton_api->registers.x86_esi);
		g_symvar_bp = triton_api->symbolizeRegister(triton_api->registers.x86_rbp, "rbp");
		g_symvar_sp = triton_api->symbolizeRegister(triton_api->registers.x86_rsp, "rsp");
	}
	else
	{
		// non-const: esp, ebp
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ebx, cpu_state.rbx);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_esi, cpu_state.rsi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_edi, cpu_state.rdi);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_ebp, cpu_state.rbp);
		triton_api->setConcreteRegisterValue(triton_api->registers.x86_esp, cpu_state.rsp);

		triton::uint64 arg0 = cpu_state.rbp;
		triton_api->setConcreteMemoryAreaValue(cpu_state.rbp, (const triton::uint8*) & arg0, 8);

		//triton_api->taintRegister(triton_api->registers.x86_esi);
		g_symvar_bp = triton_api->symbolizeRegister(triton_api->registers.x86_ebp, "ebp");
		g_symvar_sp = triton_api->symbolizeRegister(triton_api->registers.x86_esp, "esp");
	}

	g_vmp_stackpointer = cpu_state.rbp;
	g_x86_stackpointer = cpu_state.rsp;

	_runtime_optimize(triton_api, stream, runtime_address, cpu_state, instructions_o);
}


void runtime_optimize(AbstractStream& stream,
	triton::uint64 address, triton::uint64 module_base, triton::uint64 section_addr, triton::uint64 section_size)
{
	g_stream = &stream;
	g_module_base = module_base;
	g_vmp0_address = section_addr;
	g_vmp0_size = section_size;
	g_vmp_section_address = g_module_base + g_vmp0_address;
	g_vmp_section_size = g_vmp0_size;

	std::list<std::shared_ptr<x86_instruction>> instructions;
	auto print_and_clear = [&instructions]()
	{
		for (auto i : instructions)
		{
			std::cout << std::left << std::hex << "\t" << std::setw(8) << i->get_addr() << " - " << i->get_string() << "\n";
		}
		instructions.clear();
	};

	triton::uint64 runtime_address = address;
	VMP_CPU_STATE cpu_state = {};
	cpu_state.x64 = stream.is_x86_64();
	_runtime_optimize_enter(stream, runtime_address, cpu_state, instructions);
	print_and_clear();

	while (runtime_address)
	{
		// rbp > rsp at least
		cpu_state.rbp = 0x2000;
		cpu_state.rsp = 0x1000;

		std::cout << "b-" << std::hex << cpu_state.rsi << "(" << runtime_address << "):\n";
		_runtime_optimize_handler(stream, runtime_address, cpu_state, instructions);
		print_and_clear();
	}

	std::map<x86_register, bool> dead_registers;
	xed_uint32_t dead_flags = 0xFFFFFFFF;
	apply_dead_store_elimination(instructions, dead_registers, dead_flags);
}