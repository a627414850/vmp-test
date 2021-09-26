#pragma once

#include "x86_operand.hpp"
#include "x86_register.hpp"

// The xed_decoded_inst_t has more information than the xed_encoder_request_t, 
// but both types are derived from a set of common fields called the xed_operand_values_t.

// The decoder has an operands array that holds order of the decoded operands. 
// This array indicates whether or not the operands are read or written.
class x86_instruction : private xed_decoded_inst_t
{
public:
	x86_instruction(unsigned long long addr = 0);
	~x86_instruction();

	void decode(const void *buf, unsigned int length, 
		xed_machine_mode_enum_t mmode, xed_address_width_enum_t stack_addr_width);
	void encode();

	// xed functions
	inline const char* get_name() const
	{
		return xed_iclass_enum_t2str(this->get_iclass());
	}
	inline xed_category_enum_t get_category() const
	{
		return xed_decoded_inst_get_category(this);
	}
	inline xed_extension_enum_t get_extension() const
	{
		return xed_decoded_inst_get_extension(this);
	}
	inline xed_isa_set_enum_t get_isa_set() const
	{
		return xed_decoded_inst_get_isa_set(this);
	}
	inline xed_iclass_enum_t get_iclass() const
	{
		return xed_decoded_inst_get_iclass(this);
	}

	// mode
	inline xed_uint_t get_machine_mode_bits() const
	{
		return xed_decoded_inst_get_machine_mode_bits(this);
	}
	inline xed_uint_t get_stack_address_mode_bits() const
	{
		return xed_decoded_inst_get_stack_address_mode_bits(this);
	}

	// bytes
	inline xed_uint_t get_length() const
	{
		return xed_decoded_inst_get_length(this);
	}
	inline xed_uint8_t get_byte(xed_uint_t byte_index) const
	{
		return xed_decoded_inst_get_byte(this, byte_index);
	}

	// operands
	inline unsigned int get_operand_length_bits(unsigned int operand_index) const
	{
		return xed_decoded_inst_operand_length_bits(this, operand_index);
	}
	inline unsigned int get_operand_length(unsigned int operand_index) const
	{
		return xed_decoded_inst_operand_length(this, operand_index);
	}
	inline unsigned int get_noperands()
	{
		return xed_decoded_inst_noperands(this);
	}


	inline xed_iform_enum_t get_iform_enum() const
	{
		return xed_decoded_inst_get_iform_enum(this);
	}

	// operands
	inline const xed_operand_values_t* operands_const()
	{
		return xed_decoded_inst_operands_const(this);
	}

	// register
	inline x86_register get_register(xed_operand_enum_t name = XED_OPERAND_REG0) const
	{
		return xed_decoded_inst_get_reg(this, name);
	}

	// memory
	inline xed_uint_t get_number_of_memory_operands() const
	{
		return xed_decoded_inst_number_of_memory_operands(this);
	}
	inline bool is_mem_read(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_mem_read(this, mem_idx) != 0;
	}
	inline bool is_mem_written(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_mem_written(this, mem_idx) != 0;
	}
	inline bool is_mem_written_only(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_mem_written_only(this, mem_idx);
	}
	inline x86_register get_segment_register(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_seg_reg(this, mem_idx);
	}
	inline x86_register get_base_register(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_base_reg(this, mem_idx);
	}
	inline x86_register get_index_register(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_index_reg(this, mem_idx);
	}
	inline xed_uint_t get_scale(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_scale(this, mem_idx);
	}
	inline xed_uint_t has_displacement() const
	{
		return xed_operand_values_has_memory_displacement(this);
	}
	inline xed_int64_t get_memory_displacement(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_memory_displacement(this, mem_idx);
	}
	inline xed_uint_t get_memory_displacement_width(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_memory_displacement_width(this, mem_idx);
	}
	inline xed_uint_t get_memory_displacement_width_bits(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_memory_displacement_width_bits(this, mem_idx);
	}
	inline xed_uint_t get_memory_operand_length(unsigned int mem_idx = 0) const
	{
		return xed_decoded_inst_get_memory_operand_length(this, mem_idx);
	}

	// branch
	inline xed_int32_t get_branch_displacement() const
	{
		return xed_decoded_inst_get_branch_displacement(this);
	}
	inline xed_uint_t get_branch_displacement_width() const
	{
		return xed_decoded_inst_get_branch_displacement_width(this);
	}
	inline xed_uint_t get_branch_displacement_width_bits() const
	{
		return xed_decoded_inst_get_branch_displacement_width_bits(this);
	}

	// immediate
	inline xed_uint_t get_immediate_width() const
	{
		return xed_decoded_inst_get_immediate_width(this);
	}
	inline xed_uint_t get_immediate_width_bits() const
	{
		return xed_decoded_inst_get_immediate_width_bits(this);
	}
	inline bool get_immediate_is_signed() const
	{
		// Return true if the first immediate (IMM0) is signed.
		return xed_decoded_inst_get_immediate_is_signed(this) == 1;
	}
	inline xed_int32_t get_signed_immediate() const
	{
		return xed_decoded_inst_get_signed_immediate(this);
	}
	inline xed_uint64_t get_unsigned_immediate() const
	{
		if (!this->get_signed_immediate())
			return xed_decoded_inst_get_unsigned_immediate(this);

		return xed_sign_extend_arbitrary_to_64(
			xed_decoded_inst_get_signed_immediate(this), this->get_immediate_width_bits());
	}
	inline xed_uint8_t get_second_immediate() const
	{
		return xed_decoded_inst_get_second_immediate(this);
	}

	// modification
	inline void set_scale(xed_uint_t scale)
	{
		xed_decoded_inst_set_scale(this, scale);
	}
	inline void set_memory_displacement(xed_int64_t disp, xed_uint_t length_bytes)
	{
		xed_decoded_inst_set_memory_displacement(this, disp, length_bytes);
	}
	inline void set_branch_displacement(xed_int32_t disp, xed_uint_t length_bytes)
	{
		xed_decoded_inst_set_branch_displacement(this, disp, length_bytes);
	}
	inline void set_immediate_signed(xed_int32_t x, xed_uint_t length_bytes)
	{
		xed_decoded_inst_set_immediate_signed(this, x, length_bytes);
	}
	inline void set_immediate_unsigned(xed_uint64_t x, xed_uint_t length_bytes)
	{
		xed_decoded_inst_set_immediate_unsigned(this, x, length_bytes);
	}
	inline void set_memory_displacement_bits(xed_int64_t disp, xed_uint_t length_bits)
	{
		xed_decoded_inst_set_memory_displacement_bits(this, disp, length_bits);
	}
	inline void set_branch_displacement_bits(xed_int32_t disp, xed_uint_t length_bits)
	{
		xed_decoded_inst_set_branch_displacement_bits(this, disp, length_bits);
	}
	inline void set_immediate_signed_bits(xed_int32_t x, xed_uint_t length_bits)
	{
		xed_decoded_inst_set_immediate_signed_bits(this, x, length_bits);
	}
	inline void set_immediate_unsigned_bits(xed_uint64_t x, xed_uint_t length_bits)
	{
		xed_decoded_inst_set_immediate_unsigned_bits(this, x, length_bits);
	}

	// encoder
	inline void encoder_init_from_decode()
	{
		xed_encoder_request_init_from_decode(this);
	}
	inline xed_iclass_enum_t encoder_get_iclass() const
	{
		xed_encoder_request_get_iclass(this);
	}
	inline void encoder_set_iclass(xed_iclass_enum_t iclass)
	{
		xed_encoder_request_set_iclass(this, iclass);
	}
	inline xed_operand_enum_t encoder_get_operand_order(xed_uint_t operand_index)
	{
		return xed_encoder_request_get_operand_order(this, operand_index);
	}
	inline void encoder_set_operand_order(xed_uint_t operand_index, xed_operand_enum_t name)
	{
		xed_encoder_request_set_operand_order(this, operand_index, name);
	}

	// encoder reg
	inline void encoder_set_reg(xed_operand_enum_t operand, xed_reg_enum_t reg)
	{
		xed_encoder_request_set_reg(this, operand, reg);
	}

	// encoder mem
	inline void encoder_set_memory_displacement(xed_int64_t memdisp, xed_uint_t nbytes)
	{
		xed_encoder_request_set_memory_displacement(this, memdisp, nbytes);
	}
	inline void encoder_set_agen()
	{
		xed_encoder_request_set_agen(this);
	}
	inline void encoder_set_mem0()
	{
		xed_encoder_request_set_mem0(this);
	}
	inline void encoder_set_mem1()
	{
		xed_encoder_request_set_mem1(this);
	}
	inline void encoder_set_memory_operand_length(xed_uint_t nbytes)
	{
		xed_encoder_request_set_memory_operand_length(this, nbytes);
	}
	inline void encoder_set_seg0(xed_reg_enum_t seg_reg)
	{
		xed_encoder_request_set_seg0(this, seg_reg);
	}
	inline void encoder_set_seg1(xed_reg_enum_t seg_reg)
	{
		xed_encoder_request_set_seg1(this, seg_reg);
	}
	inline void encoder_set_base0(xed_reg_enum_t base_reg)
	{
		xed_encoder_request_set_base0(this, base_reg);
	}
	inline void encoder_set_base1(xed_reg_enum_t base_reg)
	{
		xed_encoder_request_set_base1(this, base_reg);
	}
	inline void encoder_set_index(xed_reg_enum_t index_reg)
	{
		xed_encoder_request_set_index(this, index_reg);
	}
	inline void encoder_set_scale(xed_uint_t scale)
	{
		xed_encoder_request_set_scale(this, scale);
	}

	// encoder imm
	inline void encoder_set_uimm0(xed_uint64_t uimm, xed_uint_t nbytes)
	{
		xed_encoder_request_set_uimm0(this, uimm, nbytes);
	}
	inline void encoder_set_uimm0_bits(xed_uint64_t uimm, xed_uint_t nbits)
	{
		xed_encoder_request_set_uimm0_bits(this, uimm, nbits);
	}
	inline void encoder_set_uimm1(xed_uint8_t uimm)
	{
		xed_encoder_request_set_uimm1(this, uimm);
	}
	inline void encoder_set_simm(xed_int32_t simm, xed_uint_t nbytes)
	{
		xed_encoder_request_set_simm(this, simm, nbytes);
	}

	// encoder branches
	inline void encoder_set_relbr()
	{
		xed_encoder_request_set_relbr(this);
	}
	inline void encoder_set_branch_displacement(xed_int32_t brdisp, xed_uint_t nbytes)
	{
		xed_encoder_request_set_branch_displacement(this, brdisp, nbytes);
	}
	inline void encoder_set_ptr()
	{
		xed_encoder_request_set_ptr(this);
	}

	// flags
	inline xed_bool_t uses_rflags() const
	{
		return xed_decoded_inst_uses_rflags(this);
	}
	inline const xed_flag_set_t* get_read_flag_set() const
	{
		const xed_simple_flag_t* rfi = xed_decoded_inst_get_rflags_info(this);
		return xed_simple_flag_get_read_flag_set(rfi);
	}
	inline const xed_flag_set_t* get_written_flag_set() const
	{
		const xed_simple_flag_t* rfi = xed_decoded_inst_get_rflags_info(this);
		return xed_simple_flag_get_written_flag_set(rfi);
	}
	inline xed_bool_t writes_flags() const
	{
		const xed_simple_flag_t* rfi = xed_decoded_inst_get_rflags_info(this);
		return rfi ? xed_simple_flag_writes_flags(rfi) : 0;
	}

	// my functions
	inline xed_uint64_t get_addr() const
	{
		return this->m_addr;
	}
	const x86_operand get_operand(unsigned int i) const &;
	std::vector<x86_operand> get_operands() const;
	std::vector<xed_uint8_t> get_bytes() const;

	void get_read_written_registers(std::vector<x86_register>* read_registers, std::vector<x86_register>* written_registers) const;
	std::vector<x86_register> get_read_registers() const;
	std::vector<x86_register> get_written_registers() const;

	// additional
	bool is_branch() const;

	// debug functions
	std::string get_string() const;
	void sprintf(char* buf, int length) const;
	void print() const;

private:
	xed_uint64_t m_addr;
	xed_uint8_t m_bytes[16];
};