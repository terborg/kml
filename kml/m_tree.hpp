/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004, 2005 by Rutger W. ter Borg                         *
 *                                                                         *
 *  This program is free software; you can redistribute it and/or          *
 *  modify it under the terms of the GNU General Public License            *
 *  as published by the Free Software Foundation; either version 2         *
 *  of the License, or (at your option) any later version.                 *
 *                                                                         *
 *  This program is distributed in the hope that it will be useful,        *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          *
 *  GNU General Public License for more details.                           *
 *                                                                         *
 *  You should have received a copy of the GNU General Public License      *
 *  along with this program; if not, write to the Free Software            *
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307  *
 ***************************************************************************/

#ifndef M_TREE_HPP
#define M_TREE_HPP



/*!

M-tree: An Efficient Access Method for Similarity Search in Metric Spaces



Two types of nodes:

- leaf nodes: store all indexed objects, either the key or the features of the data

- internal nodes: store so-called routing objects. It has an associated pointer to the
  root of a subtree called the covering tree of the routing object. It also keeps a 
  covering radius, and the distance and pointer to its parent node. 









*/







namespace kml {


class m_tree {





	/*! Insert a point into the M-tree. */
	void insert( ... ) {
	
		// search a suitable routing node
		
		
		// the M-tree grows bottom-up
		
		
	
	
	
	
	
	
	}












}


} //namespace kml

#endif

