#pragma once

#include "AbstractStream.hpp"

struct BasicBlock;
class vtilblk;

struct VMPHandlerContext
{
	// before start
	triton::uint64 scratch_area_size;
	triton::uint64 address;
	triton::uint64 vmp_sp, bytecode, x86_sp;
	triton::engines::symbolic::SharedSymbolicVariable symvar_vmp_sp, symvar_bytecode, symvar_x86_sp;

	// load
	std::map<triton::usize, triton::engines::symbolic::SharedSymbolicVariable> scratch_variables, arguments;

	// expressions
	std::map<triton::usize, vtil::operand> expression_map; // associate symbolic variable with IR::Expression
};

class VMProtectAnalyzer
{
public:
	VMProtectAnalyzer(triton::arch::architecture_e arch = triton::arch::ARCH_X86);
	~VMProtectAnalyzer();

	// helpers
	bool is_x64() const;

	triton::arch::Register get_bp_register() const;
	triton::arch::Register get_sp_register() const;
	triton::arch::Register get_ip_register() const;

	triton::uint64 get_bp() const;
	triton::uint64 get_sp() const;
	triton::uint64 get_ip() const;

	// lea ast
	bool is_bytecode_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context);
	bool is_stack_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context);
	bool is_scratch_area_address(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context);
	bool is_fetch_arguments(const triton::ast::SharedAbstractNode &lea_ast, VMPHandlerContext *context);

	// work
	void load(AbstractStream& stream,
		unsigned long long module_base, unsigned long long vmp0_address, unsigned long long vmp0_size);

	// vm-enter
	std::map<triton::usize, vtil::register_desc> symbolize_registers();
	void analyze_vm_enter(AbstractStream& stream, triton::uint64 address);


	// vm-handler
	void symbolize_memory(const triton::arch::MemoryAccess& mem, VMPHandlerContext* context);
	std::vector<vtil::operand> save_expressions(triton::arch::Instruction& triton_instruction, VMPHandlerContext* context);
	void check_arity_operation(triton::arch::Instruction& triton_instruction, const std::vector<vtil::operand>& vtil_operands, VMPHandlerContext* context);
	void check_store_access(triton::arch::Instruction& triton_instruction, VMPHandlerContext* context);
	void modify_sp(VMPHandlerContext* context);
	void analyze_vm_handler(AbstractStream& stream, triton::uint64 handler_address);


	// etc
	void print_output();

private:
	std::shared_ptr<triton::API> triton_api;

	// after vm_enter
	unsigned long long m_scratch_size;

	// runtimeshit
	std::map<triton::uint64, std::shared_ptr<BasicBlock>> m_handlers;

	vtil::basic_block* m_block;
};