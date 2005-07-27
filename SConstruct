
import cpuinfo, os

env = Environment()

print "Platform:", env['PLATFORM']
print "Compiler:", env['CXX']

#
# Start with a search for dependencies
#

# Start with the default include path
cpp_path = ['#']

# Search path dictionaries for dependent libraries
boost_search_path = { 'posix': ['/usr/include/boost'],
                      'win32': ['/boost'] }
atlas_search_path = { 'posix': ['/usr/include', '/usr/include/atlas' ],
                      'win32': ['/atlas','/atlas/include'] }


path = env.FindFile( 'version.hpp', boost_search_path[ env['PLATFORM'] ] )
if path == None:
	print "Could not find Boost! Please check installation and/or search path in SConstruct file."
	Exit(1)
else:
	# remove the boost directory part from the directory
	boost_path = os.path.split(os.path.dirname(path.abspath))[0]
	if not cpp_path.count( boost_path ):
		cpp_path.append( boost_path )

path = env.FindFile( 'cblas.h', atlas_search_path[ env['PLATFORM'] ] )
if path == None:
	print "Could not find ATLAS! Please check installation and/or search path in SConstruct file."
	Exit(1)
else:
	atlas_path = os.path.dirname(path.abspath)
	if not cpp_path.count( atlas_path ):
		cpp_path.append( atlas_path )
   
# Adjust the environment
env.Replace( CPPPATH = cpp_path )



#
# Alright, perform some basic compile tests (of available libraries etc.)
#

#conf = Configure(env)

# First, make sure some prerequisites have been met
#if not conf.CheckCHeader('atlas/cblas.h'):
#	print 'atlas/cblas.h not found! Please check ATLAS installation.'
#	Exit(1)

#env = conf.Finish()



cpu = cpuinfo.cpuinfo()

cc_flags = ''
debug_flags = ''
optimise_flags = ''

if env['CXX'] == 'g++':
        # change default CXXflags to something which is 
	# CXXFLAGS = ....
	cc_flags += '-Wall -ansi -pedantic'
	optimise_flags += ' -O3 -ffast-math -fomit-frame-pointer'
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

elif env['CXX'] == 'msvc':
        cc_flags += '/Wall'
	optimise_flags += '/O2 /Ot'
	if cpu.is_PentiumPro() or cpu.is_PentiumII() or cpu.is_pentiumIII():
		optimize_flags += ' /G6'
	if cpu.is_PentiumIV() or cpu.is_Athlon():
   		optimise_flags += ' /G7'
	if cpu.has_sse2():
		optimise_flags += ' /arch:SSE2'
	elif cpu.has_sse():
		optimise_flags += ' /arch:SSE'



# Deligate to build scripts
env.Replace( CXXFLAGS = cc_flags + ' ' + optimise_flags )
Export( 'env' )
SConscript( dirs=['example'] )



