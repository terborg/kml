#
#  The Kernel-Machine Library
#  Copyright (C) 2013 by Rutger ter Borg
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
#
 
#
# Set the project name
#
project( "kml" )

#
# Require CMake version
#
cmake_minimum_required( VERSION 2.8 )

#
# Require proper C++:
#
# -Wall:           enables all available warnings
# -std=c++98       set the C++ standard
# -pedantic:       demand strict ISO C++; reject extensions
#

add_definitions( "-Wall -std=c++98 -pedantic -Wno-long-long" )

#
# Find and setup the required packages used in this project:
# * A couple of the Boost Libraries
# * BLAS
#  
find_package( Boost REQUIRED program_options serialization thread )
find_package( BLAS )

include_directories( . )
add_subdirectory( example )


