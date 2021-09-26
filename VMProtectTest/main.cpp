#include "pch.h"

#pragma comment(lib, "xed.lib")
#pragma comment(lib, "triton.lib")

#include "VMProtectAnalyzer.hpp"
#include "ProcessStream.hpp"


void test_x86()
{
	ProcessStream stream(false);
	if (!stream.open("devirtualizeme32_vmp_3.0.9_v1.exe"))
		throw std::runtime_error("stream.open failed.");

	unsigned long long module_base = 0x00400000;

	VMProtectAnalyzer analyzer(triton::arch::ARCH_X86);
	analyzer.load(stream, module_base, 0x17000, 0x86CB0);		// vmp0

	analyzer.analyze_vm_enter(stream, 0x0040C890);

	triton::uint64 handler_address = analyzer.get_ip();
	while (handler_address)
	{
		std::cout << "start analyzing " << std::hex << handler_address << "\n";
		analyzer.analyze_vm_handler(stream, handler_address);
		std::cout << "done.\n\n\n\n";
		handler_address = analyzer.get_ip();
	}

	analyzer.print_output();
}


void test_x86_64()
{
	ProcessStream stream(true);
	if (!stream.open("devirtualizeme64_vmp_3.0.9_v1.exe"))
		throw std::runtime_error("stream.open failed.");

	unsigned long long module_base = 0x140000000ull;

	VMProtectAnalyzer analyzer(triton::arch::ARCH_X86_64);
	analyzer.load(stream, module_base, 0x1C000, 0xE1F74);		// vmp0
	analyzer.load(stream, module_base, 0x1B000, 0xA80);			// pdata

	//analyzer.analyze_vm_enter(stream, 0x1400FD439);
	//analyzer.analyze_vm_enter(stream, 0x1400FD443);
	analyzer.analyze_vm_enter(stream, 0x1400FD44D);
	//analyzer.analyze_vm_enter(stream, 0x1400FD457ull); // after messagebox

	triton::uint64 handler_address = analyzer.get_ip();
	while (handler_address)
	{
		std::cout << "start analyzing " << std::hex << handler_address << "\n";
		analyzer.analyze_vm_handler(stream, handler_address);
		std::cout << "done.\n\n\n\n";
		handler_address = analyzer.get_ip();
	}

	analyzer.print_output();
}


extern void runtime_optimize(AbstractStream& stream,
	triton::uint64 address, triton::uint64 module_base, triton::uint64 section_addr, triton::uint64 section_size);
void t()
{
	constexpr bool x86_64 = 0;
	ProcessStream stream(x86_64);
	if (x86_64)
	{
		if (!stream.open("devirtualizeme64_vmp_3.0.9_v1.exe"))
			throw std::runtime_error("stream.open failed.");

		runtime_optimize(stream, 0x1400FD439ull, 0x140000000ull, 0x1B000, 0xA80);
	}
	else
	{
		if (!stream.open("devirtualizeme32_vmp_3.0.9_v1.exe"))
			throw std::runtime_error("stream.open failed.");

		runtime_optimize(stream, 0x4312d7, 0x00400000, 0x17000, 0x86CB0);
		//runtime_optimize(stream, 0x00FA02C0, 0x00F90000, 0x1A000, 0x851A0);
		//runtime_optimize(stream, 0x00FD8A99, 0x00F90000, 0x1A000, 0x851A0);
	}
}


int main()
{
	// Once, before using Intel XED, you must call xed_tables_init() to initialize the tables Intel XED uses for encoding and decoding:
	xed_tables_init();

	try
	{
		//test_demo();
		//triton_test();
		//test_x86();
		test_x86_64();
		//t();
	}
	catch (const std::exception &ex)
	{
		std::cout << "ex: " << ex.what() << std::endl;
	}
	return 0;
}