#include "pch.h"

#include "vtilblk.hpp"
#pragma comment(linker, "/STACK:67108864")

static constexpr vtil::register_desc FLAG_CF = vtil::REG_FLAGS.select(1, 0);
static constexpr vtil::register_desc FLAG_PF = vtil::REG_FLAGS.select(1, 2);
static constexpr vtil::register_desc FLAG_AF = vtil::REG_FLAGS.select(1, 4);
static constexpr vtil::register_desc FLAG_ZF = vtil::REG_FLAGS.select(1, 6);
static constexpr vtil::register_desc FLAG_SF = vtil::REG_FLAGS.select(1, 7);
static constexpr vtil::register_desc FLAG_DF = vtil::REG_FLAGS.select(1, 10);
static constexpr vtil::register_desc FLAG_OF = vtil::REG_FLAGS.select(1, 11);

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

vtilblk::vtilblk()
{
	this->blk = vtil::basic_block::begin(0x1337);
}
vtilblk::~vtilblk()
{

}

vtil::operand vtilblk::load_operand(triton::usize symid)
{
	return operands[symid];
}

void vtilblk::load_reg(uint8_t offset, uint8_t size, triton::usize symid)
{
	auto vm_reg = make_virtual_register(offset, size);
	auto tmp = this->blk->tmp(64);
	blk->mov(tmp, vm_reg);
	this->operands[symid] = tmp;
}
void vtilblk::load_stack(uint64_t offset, triton::usize symid)
{
	auto tmp = this->blk->tmp(64);
	blk->ldd(tmp, vtil::REG_SP, offset);
	this->operands[symid] = tmp;
}
void vtilblk::load_var(triton::usize symid1, triton::usize symid2)
{
	// tmp = [op(symid1) + 0]
	auto tmp = this->blk->tmp(64);
	blk->ldd(tmp, this->load_operand(symid1), 0);
	this->operands[symid2] = tmp;
}

void vtilblk::shift_sp(int64_t offset)
{
	blk->shift_sp(offset);
}
void vtilblk::pop()
{
	// lol
	blk->shift_sp(8);
}

void vtilblk::store_reg(uint8_t offset, uint8_t size, triton::usize symid)
{
	blk->mov(make_virtual_register(offset, size), this->load_operand(symid));
}
void vtilblk::store_stack(uint8_t offset)
{
	//blk->str(vtil::REG_SP, offset, source_operand);
}

void vtilblk::unary_op()
{
	auto tmp = this->blk->tmp(64);
	//blk->mov(tmp, lhs);
}
void vtilblk::bin_op()
{
	//auto tmp = this->blk->tmp(64);

	//blk->mov(tmp, lhs)
		//->add(tmp, rhs);
}

void vtilblk::dump()
{
	vtil::logger::log(":: Before:\n");
	vtil::debug::dump(blk->owner);

	vtil::logger::log("\n");

	// executes all optimization passes
	vtil::optimizer::apply_each<
		vtil::optimizer::profile_pass,
		vtil::optimizer::collective_cross_pass
	>{}(blk->owner);

	vtil::logger::log("\n");

	vtil::logger::log(":: After:\n");
	vtil::debug::dump(blk->owner);
}