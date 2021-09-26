#pragma once

namespace triton
{
	//typedef size_t usize;
}

class vtilblk
{
public:
	vtilblk();
	~vtilblk();

	void label_begin(vtil::vip_t vip)
	{
		this->blk->label_begin(vip);
	}
	void label_end()
	{
		this->blk->label_end();
	}

	vtil::operand load_operand(triton::usize symid);

	// mov(tmp, vm_reg)
	void load_reg(uint8_t offset, uint8_t size, triton::usize symid);

	// ldd(tmp, REG_SP, offset)
	void load_stack(uint64_t offset, triton::usize symid);

	// ldd(sym2, sym1, 0)
	void load_var(triton::usize symid1, triton::usize symid2);

	void shift_sp(int64_t offset);
	void pop();

	// mov(vm_reg, source)
	void store_reg(uint8_t offset, uint8_t size, triton::usize symid);
	void store_stack(uint8_t offset);

	void unary_op();
	void bin_op();

	void dump();

private:
	vtil::basic_block* blk;
	std::map<triton::usize, vtil::register_desc> operands;
};