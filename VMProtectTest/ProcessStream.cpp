#include "pch.h"

#include "ProcessStream.hpp"
#include <Psapi.h>

ProcessStream::ProcessStream(bool x86_64) : AbstractStream(x86_64)
{
	this->m_processId = 0;
	this->m_processHandle = NULL;
	this->m_pos = 0;
}
ProcessStream::~ProcessStream()
{
	this->close();
}

bool ProcessStream::isOpen() const
{
	return this->m_processHandle != NULL;
}

bool ProcessStream::open(unsigned long pid)
{
	this->close();
	this->m_processId = pid;
	this->m_processHandle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
	return this->isOpen();
}

bool ProcessStream::open(const std::string& process_name)
{
	// close before open
	this->close();

	// Get the list of process identifiers.
	DWORD aProcesses[1024], cbNeeded;
	if (!EnumProcesses(aProcesses, sizeof(aProcesses), &cbNeeded))
	{
		return false;
	}

	// Calculate how many process identifiers were returned.
	DWORD cProcesses = cbNeeded / sizeof(DWORD);
	for (unsigned int i = 0; i < cProcesses; i++)
	{
		if (aProcesses[i] == 0)
			continue;

		// Get a handle to the process.
		HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, aProcesses[i]);
		if (hProcess == NULL)
			continue;

		CHAR szProcessName[MAX_PATH];
		if (GetModuleBaseNameA(hProcess, NULL, szProcessName, MAX_PATH) != 0
			&& process_name.compare(szProcessName) == 0)
		{
			// just break after close handle
			this->open(aProcesses[i]);
			i = cProcesses;
		}
		CloseHandle(hProcess);
	}

	return this->isOpen();
}

void ProcessStream::close()
{
	if (this->m_processHandle != NULL)
	{
		CloseHandle(this->m_processHandle);
		this->m_processHandle = NULL;
	}
}

SIZE_T ProcessStream::read(void* buf, SIZE_T size)
{
	if (!this->isOpen())
		throw std::runtime_error("process is not open");

	LPCVOID address = reinterpret_cast<LPCVOID>(this->m_pos);
	SIZE_T readBytes = 0;
	if (!ReadProcessMemory(this->m_processHandle, address, buf, size, &readBytes))
	{
		return 0;
		DWORD lastError = GetLastError();
		std::stringstream ss;
		ss << "ReadProcessMemory(" << address << ") failed with error code: " << lastError;
		throw std::runtime_error(ss.str());
	}

	this->m_pos += readBytes;
	return readBytes;
}

SIZE_T ProcessStream::write(const void* buf, SIZE_T size)
{
	if (!this->isOpen())
		throw std::runtime_error("process is not open");

	LPVOID address = reinterpret_cast<LPVOID>(this->m_pos);
	SIZE_T writtenBytes = 0;
	if (!WriteProcessMemory(this->m_processHandle, address, buf, size, &writtenBytes))
	{
		DWORD lastError = GetLastError();
		std::stringstream ss;
		ss << "WriteProcessMemory failed with error code: " << lastError;
		throw std::runtime_error(ss.str());
	}

	this->m_pos += writtenBytes;
	return writtenBytes;
}

unsigned long long ProcessStream::pos()
{
	return this->m_pos;
}

void ProcessStream::seek(unsigned long long pos)
{
	if (!this->isOpen())
		throw std::runtime_error("process is not open");

	this->m_pos = pos;
}