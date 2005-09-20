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


Two types of entities are used: Nodes and Objects.

Two types of nodes:

- leaf nodes: store all indexed objects, either the key or the features of the data

- internal nodes: store so-called routing objects. It has an associated pointer to the
  root of a subtree called the covering tree of the routing object. It also keeps a 
  covering radius, and the distance and pointer to its parent node. 

In other words:

Lead nodes keep the vast majority of objects in the tree, whereas the internal nodes keep one
single object for routing reasons. 

Split strategy that I will use: Gonzales' algorithm


*/


#include <boost/graph/adjacency_list.hpp>


namespace kml {



class node {

	typedef int PointDescriptor;

	// a node contains at least one object (or exactly one, to be determined).
	
	// --> covering radius
	// --> distance to parent
	
};





class m_tree {

	// define an internal property map?



	/*! Insert a point into the M-tree. */
	
	// DataMap is our data
	void insert() { // VertexDescriptor point, DataMap map ) {
	
		// do we have a root node?
		
		if (!root) {
			
			root = add_vertex( my_graph );
		} else {
			
			// find a suitable 
		
		
		
		
		}
	}


	
	void range () {
		// BFS visitor ! ! ! ! 
	}
	
	
	
	// split policy: Gonzales
	void split( ) {

	
	
	}
	
	
	
	


	typedef boost::adjacency_list< boost::vecS,   // the way the out-edges are stored
	                               boost::vecS    // the way the vertex set is stored
				       > graph_type;
	
	graph_type my_graph;

	// the root_node
	boost::graph_traits< graph_type >::vertex_descriptor root;

};






} //namespace kml

#endif

