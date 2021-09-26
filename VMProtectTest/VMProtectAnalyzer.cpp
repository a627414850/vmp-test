#include "pch.h"

#include "VMProtectAnalyzer.hpp"
#include "x86_instruction.hpp"
#include "AbstractStream.hpp"
#include "CFG.hpp"


constexpr bool cout_vm_enter_instructions = 1;


static constexpr vtil::register_desc make_virtual_register(uint8_t context_offset, uint8_t size)
{
	fassert(((context_offset & 7) + size) <= 8 && size);

	return {
		vtil::register_virtual,
		(size_t)context_offset / 8,
		size * 8,
		(context_offset % 8) * 8
	};
}

// helper?
void print_basic_blocks(const std::shared_ptr<BasicBlock> &first_basic_block)
{
	std::set<unsigned long long> visit_for_print;
	std::shared_ptr<BasicBlock> basic_block = first_basic_block;
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const auto& instruction = *it;
		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			instruction->print();
			continue;
		}

		// dont print unconditional jmp, they are annoying
		if (instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| instruction->get_branch_displacement_width() == 0)
		{
			instruction->print();
		}

		visit_for_print.insert(basic_block->leader);
		if (basic_block->next_basic_block && visit_for_print.count(basic_block->next_basic_block->leader) <= 0)
		{
			// print next
			basic_block = basic_block->next_basic_block;
		}
		else if (basic_block->target_basic_block && visit_for_print.count(basic_block->target_basic_block->leader) <= 0)
		{
			// it ends with jmp?
			basic_block = basic_block->target_basic_block;
		}
		else
		{
			// perhaps finishes?
			break;
		}

		it = basic_block->instructions.begin();
	}
}

// variablenode?
triton::engines::symbolic::SharedSymbolicVariable get_symbolic_var(const triton::ast::SharedAbstractNode &node)
{
	return node->getType() == triton::ast::VARIABLE_NODE ? 
		std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable() : nullptr;
}
std::set<triton::ast::SharedAbstractNode> collect_symvars(const triton::ast::SharedAbstractNode &node)
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
bool is_unary_operation(const triton::arch::Instruction &triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_INC:
		case triton::arch::x86::ID_INS_DEC:
		case triton::arch::x86::ID_INS_NEG:
		case triton::arch::x86::ID_INS_NOT:
			return true;

		default:
			return false;
	}
}
bool is_binary_operation(const triton::arch::Instruction &triton_instruction)
{
	switch (triton_instruction.getType())
	{
		case triton::arch::x86::ID_INS_ADD:
		case triton::arch::x86::ID_INS_SUB:
		case triton::arch::x86::ID_INS_SHL:
		case triton::arch::x86::ID_INS_SHR:
		case triton::arch::x86::ID_INS_RCR:
		case triton::arch::x86::ID_INS_RCL:
		case triton::arch::x86::ID_INS_ROL:
		case triton::arch::x86::ID_INS_ROR:
		case triton::arch::x86::ID_INS_AND:
		case triton::arch::x86::ID_INS_OR:
		case triton::arch::x86::ID_INS_XOR:
		//case triton::arch::x86::ID_INS_CMP:
		//case triton::arch::x86::ID_INS_TEST:
		case triton::arch::x86::ID_INS_MUL:
		case triton::arch::x86::ID_INS_IMUL:
			return true;

		default:
			return false;
	}
}


// VMProtectAnalyzer
VMProtectAnalyzer::VMProtectAnalyzer(triton::arch::architecture_e arch)
{
	triton_api = std::make_shared<triton::API>();
	triton_api->setArchitecture(arch);
	triton_api->setMode(triton::modes::ALIGNED_MEMORY, true);
	triton_api->setMode(triton::modes::CONSTANT_FOLDING, true);
	//triton_api->setAstRepresentationMode(triton::ast::representations::PYTHON_REPRESENTATION);
	this->m_scratch_size = 0;
}
VMProtectAnalyzer::~VMProtectAnalyzer()
{
}

bool VMProtectAnalyzer::is_x64() const
{
	const triton::arch::architecture_e architecture = this->triton_api->getArchitecture();
	switch (architecture)
	{
		case triton::arch::ARCH_X86:
			return false;

		case triton::arch::ARCH_X86_64:
			return true;

		default:
			throw std::runtime_error("invalid architecture");
	}
}

triton::arch::Register VMProtectAnalyzer::get_bp_register() const
{
	return this->is_x64() ? triton_api->registers.x86_rbp : triton_api->registers.x86_ebp;
}
triton::arch::Register VMProtectAnalyzer::get_sp_register() const
{
	const triton::arch::CpuInterface *_cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getStackPointer();
}
triton::arch::Register VMProtectAnalyzer::get_ip_register() const
{
	const triton::arch::CpuInterface *_cpu = triton_api->getCpuInstance();
	if (!_cpu)
		throw std::runtime_error("CpuInterface is nullptr");

	return _cpu->getProgramCounter();
}

triton::uint64 VMProtectAnalyzer::get_bp() const
{
	return triton_api->getConcreteRegisterValue(this->get_bp_register()).convert_to<triton::uint64>();
}
triton::uint64 VMProtectAnalyzer::get_sp() const
{
	return triton_api->getConcreteRegisterValue(this->get_sp_register()).convert_to<triton::uint64>();
}
triton::uint64 VMProtectAnalyzer::get_ip() const
{
	return triton_api->getConcreteRegisterValue(this->get_ip_register()).convert_to<triton::uint64>();
}

bool VMProtectAnalyzer::is_bytecode_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	// return true if lea_ast is constructed by bytecode
	const std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(lea_ast);
	if (symvars.empty())
		return false;

	for (auto it = symvars.begin(); it != symvars.end(); ++it)
	{
		const triton::ast::SharedAbstractNode &node = *it;
		const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
		if (symvar->getId() != context->symvar_bytecode->getId())
			return false;
	}
	return true;
}
bool VMProtectAnalyzer::is_stack_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	// return true if lea_ast is constructed by stack
	const std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(lea_ast);
	if (symvars.empty())
		return false;

	for (auto it = symvars.begin(); it != symvars.end(); ++it)
	{
		const triton::ast::SharedAbstractNode &node = *it;
		const triton::engines::symbolic::SharedSymbolicVariable &symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(node)->getSymbolicVariable();
		if (symvar != context->symvar_vmp_sp)
			return false;
	}
	return true;
}
bool VMProtectAnalyzer::is_scratch_area_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	// size is hardcoded for now (can see in any push handler perhaps)
	const triton::uint64 runtime_address = lea_ast->evaluate().convert_to<triton::uint64>();
	return context->x86_sp <= runtime_address && runtime_address < (context->x86_sp + context->scratch_area_size);
}
bool VMProtectAnalyzer::is_fetch_arguments(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context)
{
	if (lea_ast->getType() != triton::ast::VARIABLE_NODE)
		return false;

	const triton::engines::symbolic::SharedSymbolicVariable &symvar =
		std::dynamic_pointer_cast<triton::ast::VariableNode>(lea_ast)->getSymbolicVariable();
	return context->arguments.find(symvar->getId()) != context->arguments.end();
}

void VMProtectAnalyzer::load(AbstractStream& stream,
	unsigned long long module_base, unsigned long long vmp0_address, unsigned long long vmp0_size)
{
	// concretize vmp section memory
	unsigned long long vmp_section_address = (module_base + vmp0_address);
	unsigned long long vmp_section_size = vmp0_size;
	void *vmp0 = malloc(vmp_section_size);

	stream.seek(vmp_section_address);
	if (stream.read(vmp0, vmp_section_size) != vmp_section_size)
		throw std::runtime_error("stream.read failed");

	triton_api->setConcreteMemoryAreaValue(vmp_section_address, (const triton::uint8 *)vmp0, vmp_section_size);
	free(vmp0);
}

// vm-enter
std::map<triton::usize, vtil::register_desc> VMProtectAnalyzer::symbolize_registers()
{
	std::map<triton::usize, vtil::register_desc> regmap;
	auto _work = [this, &regmap](const triton::arch::Register& reg, x86_reg value)
	{
		auto symvar = triton_api->symbolizeRegister(reg);
		symvar->setAlias(reg.getName());
		regmap.insert(std::make_pair(symvar->getId(), vtil::register_cast<x86_reg>{}(value)));
	};
	if (this->is_x64())
	{
		_work(triton_api->registers.x86_rax, X86_REG_RAX);
		_work(triton_api->registers.x86_rbx, X86_REG_RBX);
		_work(triton_api->registers.x86_rcx, X86_REG_RCX);
		_work(triton_api->registers.x86_rdx, X86_REG_RDX);
		_work(triton_api->registers.x86_rsi, X86_REG_RSI);
		_work(triton_api->registers.x86_rdi, X86_REG_RDI);
		_work(triton_api->registers.x86_rbp, X86_REG_RBP);
		_work(triton_api->registers.x86_r8, X86_REG_R8);
		_work(triton_api->registers.x86_r9, X86_REG_R9);
		_work(triton_api->registers.x86_r10, X86_REG_R10);
		_work(triton_api->registers.x86_r11, X86_REG_R11);
		_work(triton_api->registers.x86_r12, X86_REG_R12);
		_work(triton_api->registers.x86_r13, X86_REG_R13);
		_work(triton_api->registers.x86_r14, X86_REG_R14);
		_work(triton_api->registers.x86_r15, X86_REG_R15);
	}
	else
	{
		_work(triton_api->registers.x86_eax, X86_REG_EAX);
		_work(triton_api->registers.x86_ebx, X86_REG_EBX);
		_work(triton_api->registers.x86_ecx, X86_REG_ECX);
		_work(triton_api->registers.x86_edx, X86_REG_EDX);
		_work(triton_api->registers.x86_esi, X86_REG_ESI);
		_work(triton_api->registers.x86_edi, X86_REG_EDI);
		_work(triton_api->registers.x86_ebp, X86_REG_EBP);
	}
	return regmap;
}
void VMProtectAnalyzer::analyze_vm_enter(AbstractStream& stream, triton::uint64 address)
{
	// reset triton api
	triton_api->clearCallbacks();
	triton_api->concretizeAllMemory();
	triton_api->concretizeAllRegister();
	auto regmap = this->symbolize_registers();

	// set esp
	const triton::arch::Register sp_register = this->get_sp_register();
	triton_api->setConcreteRegisterValue(sp_register, 0x1000);

	const triton::uint64 previous_sp = this->get_sp();
	bool check_flags = true;

	std::shared_ptr<BasicBlock> basic_block = make_cfg(stream, address);
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();

		// concrete ip as some instruction read (E|R)IP
		triton_api->setConcreteRegisterValue(this->get_ip_register(), xed_instruction->get_addr());

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());
		if (!triton_api->processing(triton_instruction))
		{
			throw std::runtime_error("triton processing failed");
		}

		// check flags
		if (check_flags)
		{
			// symbolize memory if pushfd or pushfq
			if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD
				|| triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
			{
				const auto& stores = triton_instruction.getStoreAccess();
				assert(stores.size() == 1);

				auto symvar_eflags = triton_api->symbolizeMemory(stores.begin()->first);
				regmap.insert(std::make_pair(symvar_eflags->getId(), vtil::REG_FLAGS));
			}

			// written_register
			for (const auto &pair : triton_instruction.getWrittenRegisters())
			{
				const triton::arch::Register &written_register = pair.first;
				if (triton_api->isFlag(written_register))
				{
					check_flags = false;
					break;
				}
			}
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			if (cout_vm_enter_instructions)
				std::cout << triton_instruction << "\n";
			continue;
		}

		if (xed_instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| xed_instruction->get_branch_displacement_width() == 0)
		{
			if (cout_vm_enter_instructions)
				std::cout << triton_instruction << "\n";
		}

		if (basic_block->next_basic_block && basic_block->target_basic_block)
		{
			// it ends with conditional branch
			if (triton_instruction.isConditionTaken())
			{
				basic_block = basic_block->target_basic_block;
			}
			else
			{
				basic_block = basic_block->next_basic_block;
			}
		}
		else if (basic_block->target_basic_block)
		{
			// it ends with jmp?
			basic_block = basic_block->target_basic_block;
		}
		else if (basic_block->next_basic_block)
		{
			// just follow :)
			basic_block = basic_block->next_basic_block;
		}
		else
		{
			// perhaps finishes?
			assert(basic_block->terminator);
			break;
		}

		it = basic_block->instructions.begin();
	}

	// create instructions
	const triton::uint64 bp = this->get_bp();
	const triton::uint64 sp = this->get_sp();
	const triton::arch::Register si_register = this->is_x64() ? triton_api->registers.x86_rsi : triton_api->registers.x86_esi;
	const triton::uint64 vip = triton_api->getConcreteRegisterValue(si_register).convert_to<triton::uint64>();
	const triton::uint64 scratch_size = bp - sp;
	const triton::uint64 scratch_length = scratch_size / triton_api->getGprSize();
	const triton::uint64 var_length = (previous_sp - bp) / triton_api->getGprSize();

	this->m_block = vtil::basic_block::begin(vip);
	for (triton::uint64 i = 0; i < var_length; i++)
	{
		triton::ast::SharedAbstractNode mem_ast = triton_api->getMemoryAst(
			triton::arch::MemoryAccess(previous_sp - (i * triton_api->getGprSize()) - triton_api->getGprSize(), triton_api->getGprSize()));
		triton::ast::SharedAbstractNode simplified = triton_api->processSimplification(mem_ast, true);
		if (!simplified->isSymbolized())
		{
			// should be immediate if not symbolized
			const triton::uint64 imm = simplified->evaluate().convert_to<triton::uint64>();
			this->m_block->push(imm);
		}
		else if (simplified->getType() == triton::ast::VARIABLE_NODE)
		{
			const triton::engines::symbolic::SharedSymbolicVariable symvar = get_symbolic_var(simplified);
			auto it = regmap.find(symvar->getId());
			if (it == regmap.end())
			{
				std::stringstream ss;
				ss << "L: " << __LINE__ << " vm enter error " << symvar;
				throw std::runtime_error(ss.str());
			}

			// push register
			this->m_block->push(it->second);
		}
		else
		{
			throw std::runtime_error("vm enter error");
		}
	}

	std::cout << "scratch_size: 0x" << std::hex << scratch_size
		<< ", scratch_length: " << std::dec << scratch_length << '\n'
		<< "handler: 0x" << std::hex << this->get_ip() << '\n';
		
	this->m_scratch_size = scratch_size;
}


// vm-handler
void VMProtectAnalyzer::symbolize_memory(const triton::arch::MemoryAccess& mem, VMPHandlerContext* context)
{
	const triton::uint64 mem_address = mem.getAddress();
	triton::ast::SharedAbstractNode lea_ast = mem.getLeaAst();
	if (!lea_ast)
	{
		// most likely can be ignored
		return;
	}

	lea_ast = triton_api->processSimplification(lea_ast, true);
	if (!lea_ast->isSymbolized())
	{
		// most likely can be ignored
		return;
	}

	if (this->is_bytecode_address(lea_ast, context))
	{
		// bytecode can be considered const value
		//triton_api->taintMemory(mem);
	}

	// lea_ast = context + const
	else if (this->is_scratch_area_address(lea_ast, context))
	{
		// the instruction loads virtual register
		const triton::uint64 context_offset = lea_ast->evaluate().convert_to<triton::uint64>() - context->x86_sp;

		triton::engines::symbolic::SharedSymbolicVariable symvar_vmreg = triton_api->symbolizeMemory(mem);
		context->scratch_variables.insert(std::make_pair(symvar_vmreg->getId(), symvar_vmreg));
		std::cout << "Load Scratch:[0x" << std::hex << context_offset << "]\n";

		// declare temp
		auto temp = this->m_block->tmp(mem.getBitSize());
		context->expression_map[symvar_vmreg->getId()] = temp;
		symvar_vmreg->setAlias(temp.to_string());

		// temp = virtual_register
		this->m_block->mov(temp, make_virtual_register(context_offset, mem.getSize()));
	}

	// 
	else if (this->is_stack_address(lea_ast, context))
	{
		const triton::uint64 offset = mem_address - context->vmp_sp;
		triton::arch::Register segment_register = mem.getConstSegmentRegister();
		if (segment_register.getId() == triton::arch::ID_REG_INVALID)
		{
			// DS?
			//segment_register = triton_api->registers.x86_ds;
		}

		triton::engines::symbolic::SharedSymbolicVariable symvar_arg = triton_api->symbolizeMemory(mem);
		context->arguments.insert(std::make_pair(symvar_arg->getId(), symvar_arg));
		std::cout << "Load [REG_SP+0x" << std::hex << offset << "]\n";

		// declare temp
		auto temp = this->m_block->tmp(mem.getBitSize());
		context->expression_map[symvar_arg->getId()] = temp;
		symvar_arg->setAlias(temp.to_string());

		// temp = [SP+offset]
		this->m_block->ldd(temp, vtil::REG_SP, offset);
	}

	//
	else if (this->is_fetch_arguments(lea_ast, context))
	{
		// lea_ast == VM_REG_X
		triton::arch::Register seg_reg = mem.getConstSegmentRegister();
		if (seg_reg.getId() == triton::arch::ID_REG_INVALID)
		{
			// DS?
			seg_reg = triton_api->registers.x86_ds;
		}
		triton::engines::symbolic::SharedSymbolicVariable symvar_source = get_symbolic_var(lea_ast);

		const triton::engines::symbolic::SharedSymbolicVariable symvar = triton_api->symbolizeMemory(mem);
		std::cout << "Deref(" << lea_ast << "," << seg_reg.getName() << ")\n";

		// IR
		auto it = context->expression_map.find(symvar_source->getId());
		if (it == context->expression_map.end())
			throw std::runtime_error("what do you mean");

		// declare Temp
		auto temp = this->m_block->tmp(mem.getBitSize());

		// Temp = memory(expr, segment, size)
		context->expression_map[symvar->getId()] = temp;
		symvar->setAlias(temp.to_string());

		// temp = [expr]
		this->m_block->ldd(temp, it->second, 0);
	}
	else
	{
		std::cout << "unknown read addr: " << std::hex << mem_address << " " << lea_ast << std::endl;
	}
}

std::vector<vtil::operand> VMProtectAnalyzer::save_expressions(triton::arch::Instruction& triton_instruction, VMPHandlerContext* context)
{
	std::vector<vtil::operand> expressions;
	if (!is_unary_operation(triton_instruction) 
		&& !is_binary_operation(triton_instruction))
	{
		return expressions;
	}

	bool do_it = false;
	auto operand_index = 0;
	if (triton_instruction.getType() == triton::arch::x86::ID_INS_MUL
		|| triton_instruction.getType() == triton::arch::x86::ID_INS_IMUL)
	{
		if (triton_instruction.operands.size() == 1)
		{
			// edx:eax = eax * r/m
			triton::arch::Register _reg = triton_api->registers.x86_eax;
			switch (triton_instruction.operands[0].getSize())
			{
				case 1: _reg = triton_api->registers.x86_al; break;
				case 2: _reg = triton_api->registers.x86_ax; break;
				case 4: _reg = triton_api->registers.x86_eax; break;
				default: throw std::runtime_error("idk whats wrong");
			}
			const auto simplified_reg = triton_api->processSimplification(triton_api->getRegisterAst(_reg), true);
			if (simplified_reg->isSymbolized())
			{
				triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(simplified_reg);
				if (!_symvar)
					throw std::runtime_error("idk whats wrong2");

				// load symbolic
				auto _it = context->expression_map.find(_symvar->getId());
				if (_it != context->expression_map.end())
				{
					expressions.push_back(_it->second);
					do_it = true;
				}
			}
			else
			{
				const triton::uint64 val = triton_api->getConcreteRegisterValue(_reg).convert_to<triton::uint64>();
				expressions.push_back(vtil::operand(val, triton_instruction.operands[0].getBitSize()));
			}
		}
		else if (triton_instruction.operands.size() == 3)
		{
			// op0 = r/m * imm
			operand_index = 1;
		}
	}

	for (; operand_index < triton_instruction.operands.size(); operand_index++)
	{
		const auto& operand = triton_instruction.operands[operand_index];
		if (operand.getType() == triton::arch::operand_e::OP_IMM)
		{
			const triton::arch::Immediate& imm = operand.getConstImmediate();
			expressions.push_back(
				vtil::operand(imm.getValue(), imm.getBitSize())
			);
		}
		else if (operand.getType() == triton::arch::operand_e::OP_MEM)
		{
			const triton::arch::MemoryAccess& _mem = operand.getConstMemory();
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getMemoryAst(_mem), true));
			if (_symvar)
			{
				// load symbolic
				auto _it = context->expression_map.find(_symvar->getId());
				if (_it != context->expression_map.end())
				{
					expressions.push_back(_it->second);
					do_it = true;
					continue;
				}
			}

			// otherwise immediate
			const triton::uint64 val = triton_api->getConcreteMemoryValue(_mem).convert_to<triton::uint64>();
			expressions.push_back(
				vtil::operand(val, operand.getBitSize())
			);
		}
		else if (operand.getType() == triton::arch::operand_e::OP_REG)
		{
			const triton::arch::Register& _reg = operand.getConstRegister();
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(triton_api->getRegisterAst(_reg), true));
			if (_symvar)
			{
				if (_symvar->getId() == context->symvar_vmp_sp->getId())
				{
					// nope...
					do_it = false;
					break;
				}

				// load symbolic
				auto _it = context->expression_map.find(_symvar->getId());
				if (_it != context->expression_map.end())
				{
					expressions.push_back(_it->second);
					do_it = true;
					continue;
				}
			}

			// otherwise immediate
			const triton::uint64 val = triton_api->getConcreteRegisterValue(_reg).convert_to<triton::uint64>();
			expressions.push_back(
				vtil::operand(val, operand.getBitSize())
			);
		}
		else
			throw std::runtime_error("invalid operand type");
	}
	if (!do_it)
		expressions.clear();
	return expressions;
}

void VMProtectAnalyzer::check_arity_operation(triton::arch::Instruction& triton_instruction, const std::vector<vtil::operand>& vtil_operands, VMPHandlerContext* context)
{
	if (triton_instruction.getType() == triton::arch::x86::ID_INS_CPUID)
	{
		throw std::runtime_error("cpuid not supported yet");
	}
	else if (triton_instruction.getType() == triton::arch::x86::ID_INS_RDTSC)
	{
		throw std::runtime_error("rdtsc not supported yet");
	}

	bool unary = is_unary_operation(triton_instruction) && vtil_operands.size() == 1;
	bool binary = is_binary_operation(triton_instruction) && vtil_operands.size() == 2;
	if (!unary && !binary)
		return;

	// mul/imul
	if ((triton_instruction.getType() == triton::arch::x86::ID_INS_MUL
		|| triton_instruction.getType() == triton::arch::x86::ID_INS_IMUL) && triton_instruction.operands.size() == 1)
	{
		// edx:eax = eax * r/m
		triton::arch::Register _reg_eax, _reg_edx;
		switch (triton_instruction.operands[0].getSize())
		{
			case 1:
			{
				_reg_eax = triton_api->registers.x86_ax;
				break;
			}
			case 2:
			{
				_reg_eax = triton_api->registers.x86_ax;
				_reg_edx = triton_api->registers.x86_dx;
				break;
			}
			case 4:
			{
				_reg_eax = triton_api->registers.x86_eax;
				_reg_edx = triton_api->registers.x86_edx;
				break;
			}
			default: throw std::runtime_error("idk whats wrong");
		}

		if (2 <= triton_instruction.operands[0].getSize())
		{
			// t0 = mul(eax, src)
			// t1 = extract(t0)
			// t2 = extract(t0)			but... is this good idea?
		}
		return;
	}

	// symbolize destination
	triton::engines::symbolic::SharedSymbolicVariable symvar;
	const auto& operand0 = triton_instruction.operands[0];
	if (operand0.getType() == triton::arch::operand_e::OP_REG)
	{
		const triton::arch::Register& _reg = operand0.getConstRegister();
		triton_api->concretizeRegister(_reg);
		symvar = triton_api->symbolizeRegister(_reg);
	}
	else if (operand0.getType() == triton::arch::operand_e::OP_MEM)
	{
		const triton::arch::MemoryAccess& _mem = operand0.getConstMemory();
		triton_api->concretizeMemory(_mem);
		symvar = triton_api->symbolizeMemory(_mem);
	}
	else
	{
		throw std::runtime_error("invalid operand type");
	}


	// unary(op0)
	auto op0 = vtil_operands[0];
	if (unary)
	{
		switch (triton_instruction.getType())
		{
			case triton::arch::x86::ID_INS_INC:
			{
				this->m_block->add(op0, 1);
				break;
			}
			case triton::arch::x86::ID_INS_DEC:
			{
				this->m_block->sub(op0, 1);
				break;
			}
			case triton::arch::x86::ID_INS_NEG:
			{
				this->m_block->neg(op0);
				break;
			}
			case triton::arch::x86::ID_INS_NOT:
			{
				this->m_block->bnot(op0);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown unary operation");
			}
		}
	}
	else
	{
		// binary
		auto op1 = vtil_operands[1];
		switch (triton_instruction.getType())
		{
			case triton::arch::x86::ID_INS_ADD:
			{
				this->m_block->add(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_SUB:
			{
				this->m_block->sub(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_SHL:
			{
				this->m_block->bshl(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_SHR:
			{
				this->m_block->bshr(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_RCR:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_RCL:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_ROL:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_ROR:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_AND:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_OR:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_XOR:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_CMP:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_TEST:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_MUL:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			case triton::arch::x86::ID_INS_IMUL:
			{
				this->m_block->bror(op0, op1);
				break;
			}
			default:
			{
				throw std::runtime_error("unknown binary operation");
			}
		}
	}

	context->expression_map[symvar->getId()] = op0;
	symvar->setAlias(op0.to_string());

	if (1)
	{
		const auto& triton_eflags = this->triton_api->registers.x86_eflags;
		auto t = this->m_block->tmp(triton_eflags.getBitSize());
		auto symvar = triton_api->symbolizeRegister(triton_eflags, t.to_string());
		context->expression_map[symvar->getId()] = t;

		this->m_block->mov(vtil::REG_FLAGS, t);
	}
}

void VMProtectAnalyzer::check_store_access(triton::arch::Instruction& triton_instruction, VMPHandlerContext* context)
{
	auto to_vtil_operand = [this, context](std::shared_ptr<triton::API> ctx, triton::ast::SharedAbstractNode node, size_t size) -> vtil::operand
	{
		if (!node->isSymbolized())
		{
			// expression is immediate
			const triton::uint64 val = node->evaluate().convert_to<triton::uint64>();
			return vtil::operand(val, size * 8);
		}

		triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(node);
		if (_symvar)
		{
			auto _it = context->expression_map.find(_symvar->getId());
			if (_it == context->expression_map.end())
			{
				throw std::runtime_error("cannot load operand from symvar");
			}
			return _it->second;
		}

		return vtil::operand();
	};

	const auto& storeAccess = triton_instruction.getStoreAccess();
	for (const std::pair<triton::arch::MemoryAccess, triton::ast::SharedAbstractNode>& pair : storeAccess)
	{
		const triton::arch::MemoryAccess& mem = pair.first;
		//const triton::ast::SharedAbstractNode &mem_ast = pair.second;
		const triton::ast::SharedAbstractNode& mem_ast = triton_api->getMemoryAst(mem);
		const triton::uint64 address = mem.getAddress();
		triton::ast::SharedAbstractNode lea_ast = mem.getLeaAst();
		if (!lea_ast)
		{
			// most likely can be ignored
			continue;
		}

		lea_ast = triton_api->processSimplification(lea_ast, true);
		if (!lea_ast->isSymbolized())
		{
			// most likely can be ignored
			continue;
		}

		if (this->is_scratch_area_address(lea_ast, context))
		{
			const triton::uint64 context_offset = lea_ast->evaluate().convert_to<triton::uint64>() - context->x86_sp;
			auto vr = make_virtual_register(context_offset, mem.getSize());
			std::cout << "Store " << vr.to_string() << '\n';

			// create IR (VM_REG = mem_ast)
			auto source_node = triton_api->processSimplification(mem_ast, true);
			triton::engines::symbolic::SharedSymbolicVariable symvar = get_symbolic_var(source_node);
			if (symvar)
			{
				auto it = context->expression_map.find(symvar->getId());
				if (it != context->expression_map.end())
				{
					// vr = op
					this->m_block->mov(vr, it->second);
				}
				else if (symvar->getId() == context->symvar_vmp_sp->getId())
				{
					// vr = sp
					this->m_block->mov(vr, vtil::REG_SP);
				}
				else
				{
					printf("%s\n", symvar->getAlias().c_str());
					throw std::runtime_error("what do you mean 2");
				}
			}
			else
			{
				std::cout << "source_node: " << source_node << std::endl;
			}
		}
		else if (this->is_stack_address(lea_ast, context))
		{
			// stores to stack
			const triton::uint64 stack_offset = address - context->vmp_sp;
			std::cout << "Store [REG_SP+0x" << std::hex << stack_offset << "]\n";

			auto simplified_source_node = triton_api->processSimplification(mem_ast, true);
			vtil::operand operand = to_vtil_operand(this->triton_api, simplified_source_node, mem.getSize());
			if (!operand.is_valid() && mem.getSize() == 2)
			{
				const triton::arch::MemoryAccess _mem(mem.getAddress(), 1);
				operand = to_vtil_operand(this->triton_api, triton_api->getMemoryAst(_mem), mem.getSize());
			}

			// should be push
			if (operand.is_valid())
			{
				// [SP+OFFSET] = expr
				this->m_block->str(vtil::REG_SP, stack_offset, operand);
			}
			else
			{
				std::cout << "unknown store addr: " << std::hex << address << ", lea_ast: " << lea_ast
					<< ", simplified_source_node: " << simplified_source_node << std::endl;
			}
		}
		else
		{
			auto simplified_source_node = triton_api->processSimplification(mem_ast, true);
			auto operand = to_vtil_operand(this->triton_api, simplified_source_node, mem.getSize());
			triton::engines::symbolic::SharedSymbolicVariable symvar0 = get_symbolic_var(lea_ast);
			if (symvar0)
			{
				auto it0 = context->expression_map.find(symvar0->getId());
				if (it0 != context->expression_map.end())
				{
					// [X+0] = expr
					this->m_block->str(it0->second, 0, operand);
				}
				else
				{
					throw std::runtime_error("what do you mean 2");
				}
			}
			else
			{
				std::cout << "unknown store addr: " << std::hex << address << ", lea_ast: " << lea_ast << ", simplified_source_node: " << simplified_source_node << std::endl;
			}
		}
	}
}

void VMProtectAnalyzer::modify_sp(VMPHandlerContext* context)
{
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::arch::Register si_register = this->is_x64() ? triton_api->registers.x86_rsi : triton_api->registers.x86_esi;
	const triton::uint64 bytecode = triton_api->getConcreteRegisterValue(si_register).convert_to<triton::uint64>();
	const triton::uint64 x86_sp = this->get_sp();
	const triton::uint64 vmp_sp = this->get_bp();

	// check x86_sp
	const triton::ast::SharedAbstractNode simplified_x86_sp =
		triton_api->processSimplification(triton_api->getRegisterAst(sp_register), true);
	std::set<triton::ast::SharedAbstractNode> symvars = collect_symvars(simplified_x86_sp);
	if (symvars.size() == 1)
	{
		const triton::ast::SharedAbstractNode _node = *symvars.begin();
		const auto _symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(_node)->getSymbolicVariable();
		if (_symvar->getId() == context->symvar_vmp_sp->getId())
		{
			// if x86_sp == compute(vmp_sp) then vm exit handler
			//this->analyze_vm_exit(context);
			return;
		}
	}

	// check vmp_sp
	const triton::ast::SharedAbstractNode simplified_vmp_sp =
		triton_api->processSimplification(triton_api->getRegisterAst(bp_register), true);
	std::cout << "REG_SP: " << simplified_vmp_sp << '\n';
	if (simplified_vmp_sp->getType() == triton::ast::BVADD_NODE)
	{
		// vmp_sp = add(vmp_sp, vmp_sp_offset)
		triton::sint64 vmp_sp_offset = this->get_bp() - context->vmp_sp;	// needs to be signed
		this->m_block->shift_sp(vmp_sp_offset);
	}
	else if (simplified_vmp_sp->getType() == triton::ast::VARIABLE_NODE)
	{
		const auto _symvar = std::dynamic_pointer_cast<triton::ast::VariableNode>(simplified_vmp_sp)->getSymbolicVariable();
		if (_symvar != context->symvar_vmp_sp)
		{
			auto it = context->expression_map.find(_symvar->getId());
			if (it == context->expression_map.end())
			{
				throw std::runtime_error("invalid vmp_sp");
			}

			this->m_block->mov(vtil::REG_SP, it->second);
		}
	}
	else
	{
		std::cout << simplified_vmp_sp << std::endl;
		throw std::runtime_error("invalid vmp_sp");
	}
}

void VMProtectAnalyzer::analyze_vm_handler(AbstractStream& stream, triton::uint64 handler_address)
{
	//this->m_scratch_size = 0xC0; // test

	// reset triton api
	triton_api->clearCallbacks();
	triton_api->concretizeAllMemory();
	triton_api->concretizeAllRegister();

	// allocate scratch area
	const triton::arch::Register bp_register = this->get_bp_register();
	const triton::arch::Register sp_register = this->get_sp_register();
	const triton::arch::Register si_register = this->is_x64() ? triton_api->registers.x86_rsi : triton_api->registers.x86_esi;
	const triton::arch::Register ip_register = this->get_ip_register();

	constexpr unsigned long c_stack_base = 0x1000;
	triton_api->setConcreteRegisterValue(bp_register, c_stack_base);
	triton_api->setConcreteRegisterValue(sp_register, c_stack_base - this->m_scratch_size);

	unsigned int arg0 = c_stack_base;
	triton_api->setConcreteMemoryAreaValue(c_stack_base, (const triton::uint8*) & arg0, 4);

	// ebp = VM's "stack" pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_vmp_sp = triton_api->symbolizeRegister(bp_register);

	// esi = pointer to VM bytecode
	triton::engines::symbolic::SharedSymbolicVariable symvar_bytecode = triton_api->symbolizeRegister(si_register);

	// x86 stack pointer
	triton::engines::symbolic::SharedSymbolicVariable symvar_x86_sp = triton_api->symbolizeRegister(sp_register);

	symvar_vmp_sp->setAlias("vmp_sp");
	symvar_bytecode->setAlias("bytecode");
	symvar_x86_sp->setAlias("x86_sp");

	// yo...
	VMPHandlerContext context;
	context.scratch_area_size = this->is_x64() ? 0x140 : 0x60;
	context.address = handler_address;
	context.vmp_sp = triton_api->getConcreteRegisterValue(bp_register).convert_to<triton::uint64>();
	context.bytecode = triton_api->getConcreteRegisterValue(si_register).convert_to<triton::uint64>();
	context.x86_sp = triton_api->getConcreteRegisterValue(sp_register).convert_to<triton::uint64>();
	context.symvar_vmp_sp = symvar_vmp_sp;
	context.symvar_bytecode = symvar_bytecode;
	context.symvar_x86_sp = symvar_x86_sp;

	// expr
	//std::shared_ptr<IR::Expression> x86_sp = std::make_shared<IR::Variable>("x86_sp", (IR::ir_size)sp_register.getSize());
	context.expression_map.insert(std::make_pair(symvar_vmp_sp->getId(), vtil::REG_SP));
	//context.expression_map.insert(std::make_pair(symvar_x86_sp->getId(), x86_sp));

	// cache basic block (maybe not best place)
	std::shared_ptr<BasicBlock> basic_block;
	auto handler_it = this->m_handlers.find(handler_address);
	if (handler_it == this->m_handlers.end())
	{
		basic_block = make_cfg(stream, handler_address);
		this->m_handlers.insert(std::make_pair(handler_address, basic_block));
	}
	else
	{
		basic_block = handler_it->second;
	}

	// insert label
	this->m_block->label_begin(context.bytecode);

	triton::uint64 expected_return_address = 0;
	for (auto it = basic_block->instructions.begin(); it != basic_block->instructions.end();)
	{
		const std::shared_ptr<x86_instruction> xed_instruction = *it;
		const std::vector<xed_uint8_t> bytes = xed_instruction->get_bytes();
		bool mem_read = false;
		for (xed_uint_t j = 0, memops = xed_instruction->get_number_of_memory_operands(); j < memops; j++)
		{
			if (xed_instruction->is_mem_read(j))
			{
				mem_read = true;
				break;
			}
		}

		// triton removes from written registers if it is NOT actually written, so xed helps here
		const bool maybe_flag_written = xed_instruction->writes_flags();

		// do stuff with triton
		triton::arch::Instruction triton_instruction;
		triton_instruction.setOpcode(&bytes[0], (triton::uint32)bytes.size());
		triton_instruction.setAddress(xed_instruction->get_addr());

		// fix ip
		triton_api->setConcreteRegisterValue(ip_register, xed_instruction->get_addr());

		// disassembly
		triton_api->disassembly(triton_instruction);
		if (mem_read
			&& (triton_instruction.getType() != triton::arch::x86::ID_INS_POP
				&& triton_instruction.getType() != triton::arch::x86::ID_INS_POPFD)) // no need but makes life easier
		{
			for (auto& operand : triton_instruction.operands)
			{
				if (operand.getType() == triton::arch::OP_MEM)
				{
					triton_api->getSymbolicEngine()->initLeaAst(operand.getMemory());
					this->symbolize_memory(operand.getConstMemory(), &context);
				}
			}
		}

		auto vtil_operands = this->save_expressions(triton_instruction, &context);
		if (!triton_api->buildSemantics(triton_instruction))
		{
			throw std::runtime_error("triton buildSemantics failed");
		}

		// works
		this->check_arity_operation(triton_instruction, vtil_operands, &context);
		this->check_store_access(triton_instruction, &context);

		if (xed_instruction->get_category() != XED_CATEGORY_UNCOND_BR
			|| xed_instruction->get_branch_displacement_width() == 0)
		{
			std::cout << "\t" << triton_instruction << "\n";
		}

		// pushfd/pushfq
		if (triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD
			|| triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFQ)
		{
			auto eflags_ast = this->triton_api->getRegisterAst(this->triton_api->registers.x86_eflags);
			triton::engines::symbolic::SharedSymbolicVariable _symvar = get_symbolic_var(triton_api->processSimplification(eflags_ast, true));
			if (_symvar)
			{
				auto it = context.expression_map.find(_symvar->getId());
				if (it == context.expression_map.end())
				{
					// ?
					throw std::runtime_error("bluh");
				}

				triton::arch::MemoryAccess _mem(this->get_sp(), triton_instruction.getType() == triton::arch::x86::ID_INS_PUSHFD ? 4 : 8);
				auto _symvar = triton_api->symbolizeMemory(_mem);
				context.expression_map[_symvar->getId()] = it->second;
			}
		}

		if (++it != basic_block->instructions.end())
		{
			// loop until it reaches end
			continue;
		}

		if (triton_instruction.getType() == triton::arch::x86::ID_INS_CALL)
		{
			expected_return_address = xed_instruction->get_addr() + 5;
		}
		else if (triton_instruction.getType() == triton::arch::x86::ID_INS_RET)
		{
			if (expected_return_address != 0 && this->get_ip() == expected_return_address)
			{
				basic_block = make_cfg(stream, expected_return_address);
				it = basic_block->instructions.begin();
			}
		}

		while (it == basic_block->instructions.end())
		{
			if (basic_block->next_basic_block && basic_block->target_basic_block)
			{
				// it ends with conditional branch
				if (triton_instruction.isConditionTaken())
				{
					basic_block = basic_block->target_basic_block;
				}
				else
				{
					basic_block = basic_block->next_basic_block;
				}
			}
			else if (basic_block->target_basic_block)
			{
				// it ends with jmp?
				basic_block = basic_block->target_basic_block;
			}
			else if (basic_block->next_basic_block)
			{
				// just follow :)
				basic_block = basic_block->next_basic_block;
			}
			else
			{
				// perhaps finishes?
				goto l_categorize_handler;
			}
			it = basic_block->instructions.begin();
		}
	}

l_categorize_handler:
	this->modify_sp(&context);
}


//
void VMProtectAnalyzer::print_output()
{
	vtil::logger::log(":: Before:\n");
	vtil::debug::dump(m_block->owner);

	vtil::logger::log("\n");

	// executes all optimization passes
	vtil::optimizer::apply_each<
		vtil::optimizer::profile_pass,
		vtil::optimizer::collective_cross_pass
	>{}(m_block->owner);

	vtil::logger::log("\n");

	vtil::logger::log(":: After:\n");
	vtil::debug::dump(m_block->owner);
}