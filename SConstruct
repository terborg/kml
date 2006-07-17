import cpuinfo, os

env = Environment()

print "Platform:", env['PLATFORM']
print "Compiler:", env['CXX']

# Start with a default cpp and lib path
cpp_path = ['#']
lib_path = []

# Improve platform dependent settings
boost_search_path = []
atlas_search_path = []
atlas_link_libs = []
lapack_link_libs = []
c_link_libs = []

if env['PLATFORM'] == 'posix':
   boost_search_path = ['/usr/include/boost']
   atlas_search_path = ['/usr/include/atlas' ]
   c_link_libs = ['kml']
   lib_path = ['/usr/lib', '/usr/lib64/atlas' ]
   atlas_link_libs = []
   if env.FindFile( 'libatlas.so.3', lib_path ): atlas_link_libs.append( 'atlas' )
   if env.FindFile( 'libcblas.so.3', lib_path ): atlas_link_libs.append( 'cblas' )
   if env.FindFile( 'liblapack_atlas.so.3', lib_path ): atlas_link_libs.append( 'lapack_atlas' )
   if env.FindFile( 'libg2c.so.0', lib_path ): atlas_link_libs.append( 'g2c' )
   if env.FindFile( 'liblapack.so.3', lib_path ): lapack_link_libs.append( 'lapack' )

elif env['PLATFORM'] == 'win32':
   env.Replace( ENV = os.environ )
   boost_search_path = ['/boost',
                        '/boost/include/boost-1_33_0/boost',
                        '/boost/include/boost-1_33_1/boost']
   atlas_search_path = ['/atlas','/atlas/include']
   # atlas_link_libs declaration is moved down to the CPU detection code part
   lib_path = ['#/lib']
   if 'VSINSTALLDIR' in os.environ:
		lib_path.append( os.environ["VSINSTALLDIR"].replace('\\','/') + '/VC/lib' )
elif env['PLATFORM'] == 'darwin':
   boost_search_path = ['/sw/include/boost']
   atlas_search_path = ['/sw/include/atlas']
   atlas_link_libs = ['cblas','atlas']
   lib_path = ['/sw/lib']
   c_link_libs = ['kml']

# Search path dictionaries for dependent libraries
path = env.FindFile( 'version.hpp', boost_search_path )
if path == None:
	print "Could not find Boost! Please check installation and/or search path in SConstruct file."
	Exit(1)
else:
	# remove the boost directory part from the directory
	boost_path = os.path.split(os.path.dirname(path.abspath))[0]
	if not cpp_path.count( boost_path ):
		cpp_path.append( boost_path )
		
# try to find the boost library	path, and add that to the global libpath
for trydir in boost_search_path:
	if os.path.exists( trydir + '/lib' ):
		boost_lib_path = trydir + '/lib'
		if not boost_lib_path in lib_path:
			lib_path += [ boost_lib_path ]
			
#path = env.FindFile( 'cblas.h', atlas_search_path )
#if path == None:
#	print "Could not find ATLAS! Please check installation and/or search path in SConstruct file."
#	Exit(1)
#else:
#	atlas_path = os.path.dirname(path.abspath)
#	if not cpp_path.count( atlas_path ):
#		cpp_path.append( atlas_path )
   
# Adjust the environment
env.Replace( CPPPATH = cpp_path )
env.Replace( LIBPATH = lib_path )



#
# Alright, perform some basic compile tests (of available libraries etc.)
#

conf = Configure(env)
global_link_libs = ['']

if env['PLATFORM'] == 'posix':
	if conf.CheckLib( 'tcmalloc' ):
		global_link_libs += ['tcmalloc']

env = conf.Finish()
env.Replace( LIBS = global_link_libs )

#
# Detect CPU type
#

cpu = cpuinfo.cpuinfo()

cc_flags = ''
debug_flags = ''
optimise_flags = ''
arch_ext = ''

if env['CXX'] == 'g++':
        # change default CXXflags to something which is 
	# CXXFLAGS = ....
	cc_flags += '-Wall -ansi -pedantic'
	optimise_flags = '-O3 -ffast-math -fomit-frame-pointer -DNDEBUG -DNO_DEBUG'
	debug_flags += ' -g'
	if cpu.is_PentiumIII():
   		optimise_flags += ' -march=pentium3'
	elif cpu.is_PentiumIV(): 
   		optimise_flags += ' -march=pentium4'
	elif cpu.is_AMDK6():
   		optimise_flags += ' -march=k6'
	elif cpu.is_AMDK6_2():
   		optimise_flags += ' -march=k6-2'
	elif cpu.is_AMDK6_3():
   		optimise_flags += ' -march=k6-3'
	elif cpu.is_Athlon():
		optimise_flags += ' -march=athlon'
	elif cpu.is_Athlon64():
		optimise_flags += ' -march=k8'
	if cpu.is_32bit() and not cpu.is_ppc():
		if cpu.has_sse() or cpu._has_sse2():
			optimise_flags += ' -mfpmath=sse'
		if cpu.has_sse():
   			optimise_flags += ' -msse'
		if cpu.has_sse2():
			optimise_flags += ' -msse2'
		if cpu.has_sse3():
			optimise_flags += ' -msse3'
		if cpu.has_3dnow():
			optimise_flags += ' -m3dnow'
		if cpu.has_mmx():
			optimise_flags += ' -mmmx'

elif env['CXX'] == '$CC' and env['CC'] == 'cl':
	# Set compiler and optimisation flags
    # for now, warnings are disabled; multithreading is enabled for correct linking
	cc_flags = '/nologo /EHsc /Zc:forScope /MT'
	optimise_flags = '/O2 /DNDEBUG /DNO_DEBUG'
	arch_ext = ''
	
	if cpu.is_PentiumIV(): 
   		arch_ext += 'P4'

	if cpu.has_sse2():
		optimise_flags += ' /arch:SSE2'
		arch_ext += 'SSE2'
	elif cpu.has_sse():
		optimise_flags += ' /arch:SSE'
		arch_ext += 'SSE'
		
	atlas_link_libs = ['cblas_' + arch_ext + '.lib']

		
	print "(future use) architecture extension: ", arch_ext

# For convenience, attach the environment's LIBS to the atlas_link_libs
atlas_link_libs += env['LIBS']
		
# Export the environment variables
Export( 'env' )
Export( 'atlas_link_libs' )
Export( 'lapack_link_libs' )
Export( 'c_link_libs' )
Export( 'arch_ext' )

# Deligate to build scripts
env.Replace( CXXFLAGS = cc_flags + ' ' + optimise_flags + debug_flags)
SConscript( dirs=['lib', 'example'] )





