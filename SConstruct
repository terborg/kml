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

if env['PLATFORM'] == 'posix':
   boost_search_path = ['/usr/include/boost']
   atlas_search_path = ['/usr/include', '/usr/include/atlas' ]
   atlas_link_libs = ['cblas', 'atlas']
elif env['PLATFORM'] == 'win32':
   env.Replace( ENV = os.environ )
   boost_search_path = ['/boost']
   atlas_search_path = ['/atlas','/atlas/include']
   atlas_link_libs = ['cblas']
   lib_path = ['#/lib']

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

path = env.FindFile( 'cblas.h', atlas_search_path )
if path == None:
	print "Could not find ATLAS! Please check installation and/or search path in SConstruct file."
	Exit(1)
else:
	atlas_path = os.path.dirname(path.abspath)
	if not cpp_path.count( atlas_path ):
		cpp_path.append( atlas_path )
   
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

if env['CXX'] == 'g++':
        # change default CXXflags to something which is 
	# CXXFLAGS = ....
	cc_flags += '-Wall -ansi -pedantic'
	optimise_flags = '-O3 -ffast-math -fomit-frame-pointer -DNDEBUG -DNO_DEBUG'
	debug_flags += ' -g -pg'
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
	if cpu.is_32bit():
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
	# /Wall show all warnings 
	# /Wp64 show warning related to future 64 bit porting
      # /GX enable exception handling - or would /EHsc be better?
      # /Zx:
	cc_flags += '/nologo /Wp64 /GX /Zc:forScope'
	
	# /02 favor speed
	# /GL is whole program optimisation
	optimise_flags = '/O2'
	if cpu.is_PentiumPro() or cpu.is_PentiumII() or cpu.is_pentiumIII():
		optimise_flags += ' /G6'
	if cpu.is_PentiumIV() or cpu.is_Athlon():
   		optimise_flags += ' /G7'
	if cpu.has_sse2():
		optimise_flags += ' /arch:SSE2'
	elif cpu.has_sse():
		optimise_flags += ' /arch:SSE'

# For convenience, attach the environment's LIBS to the atlas_link_libs
atlas_link_libs += env['LIBS']
		
# Export the environment variables
Export( 'env' )
Export( 'atlas_link_libs' )

# Deligate to build scripts
env.Replace( CXXFLAGS = cc_flags + ' ' + optimise_flags)
SConscript( dirs=['example', 'lib','test'] )
