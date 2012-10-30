// Contents of DLLDefines.h
#ifndef _DLLDEFINES_H_
#define _DLLDEFINES_H_

// #ifdef _MSC_VER
//# pragma warning(disable:4251) // warning C4251: 'member' : class 'std::???<...>' needs to have dll-interfacee to be used by clients of class
//#endif

/* Cmake will define test_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define test_EXPORTS when
building a DLL on windows.
*/
// We are using the Visual Studio Compiler and building Shared libraries


	#if defined (_WIN32)
	  #if defined(armanpytest_EXPORTS)
		#define DLLEXPORT __declspec(dllexport)
		#define TMPEXPORT
	  #else
		#define DLLEXPORT __declspec(dllimport)
		#define TMPEXPORT extern
	  #endif /* pcsim_EXPORTS */
	#else /* defined (_WIN32) */
		#define DLLEXPORT
		#define TMPEXPORT
	#endif

#endif /* _DLLDEFINES_H_ */
