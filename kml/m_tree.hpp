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

--> internal nodes do not store objects, only leaf nodes do. 2 types of nodes?
  
  

New type of tree proposed? 

Hierarchical gonzales
--> Anchors hierarchy?? (can insert, remove operations be implemented in this kind?)

*/


#include <boost/graph/adjacency_list.hpp>


namespace kml {



class node {

	typedef int PointDescriptor;
	
	// a centroid is not present in the M-tree's routing object kind of node.
	// but how to determine the distance in a query to a routing object?
	// centroid could be the "first" edge?
/*	PointDiscriptor centroid;
	double radius;	*/
	
	// can be empty?
/*	std::vector< PointDescriptor > points;*/
};





class m_tree {

	// define an internal property map?

	// covering radius of a node
	// distance to parent of the node! 


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
	
	
	
	void split( ) {
		// basically peform a 2-center on the data of the node that will be split,
		// forming 2 seperate nodes. These nodes are then linked by a new parent node.
		// but how to reference that parent node?   
	
	
	
	
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

