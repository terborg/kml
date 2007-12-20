/***************************************************************************
 *  The Kernel-Machine Library                                             *
 *  Copyright (C) 2004--2006 by Rutger W. ter Borg                         *
 *                                                                         *
 *  This library is free software; you can redistribute it and/or          *
 *  modify it under the terms of the GNU Lesser General Public             *
 *  License as published by the Free Software Foundation; either           *
 *  version 2.1 of the License, or (at your option) any later version.     *
 *                                                                         *
 *  This library is distributed in the hope that it will be useful,        *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of         *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU      *
 *  Lesser General Public License for more details.                        *
 *                                                                         *
 *  You should have received a copy of the GNU Lesser General Public       *
 *  License along with this library; if not, write to the Free Software    *
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  *
 ***************************************************************************/

#ifndef KML_M_TREE_HPP
#define KML_M_TREE_HPP

//#include <kml/tree.hh>

// for now, consider Eucledian distance
#include <kml/distance.hpp>

#include <iostream>

/*!

M-tree: An Efficient Access Method for Similarity Search in Metric Spaces


Two types of entities are used: Nodes and object.

Two types of nodes:

- leaf nodes: store all indexed object, either the key or the features of the data

- internal nodes: store so-called routing object. It has an associated pointer to the
  root of a subtree called the covering tree of the routing object. It also keeps a 
  covering radius, and the distance and pointer to its parent node. 

In other words:

Lead nodes keep the vast majority of object in the tree, whereas the internal nodes keep one
single object for routing reasons. 

Split strategy that I will use: Gonzales' algorithm


The m_tree is capable of doing the following:

It maps a key_value (that from the data container) to
-# weight? in an edge?
-# 

It also allows for a search given certain criteria.

This m_tree keeps a reference to the container, so the lifetime of the container must be the same to that of this tree.


--> good doc 
http://www.boost.org/libs/graph/doc/bundles.html

// distance to parent? 

\section References

-# M-tree: An Efficient Access Method for Similarity Search in Metric Spaces
-# Revisiting the M-tree
-# The Slim-Tree
-# The PM-tree
-# The Anchor's Hierarchy

*/


#include <boost/bind.hpp>
#include <boost/property_map.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/graph/graph_utility.hpp>

#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>



#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <utility>


namespace kml {


template<typename PropertyMap>
class m_tree {
public:
	typedef PropertyMap map_type;
	typedef m_tree<map_type> this_type;
	typedef typename boost::property_traits<PropertyMap>::key_type key_type;
	typedef typename boost::property_traits<PropertyMap>::value_type object_type;
	
	// define the node structure
	struct node {
		
		// clear the radius on construction
		node(): radius(0) {}

		// the distance to some other points could very well be 0
		// so, either the sorting on the object (close-by to far-away) should be 
		// done by excluding the centroid, or a separate entry to the centroid
		// should be kept.
		key_type centroid;
	
		// type should be deducted somehow! (value_type of the object_type? 
		// -- no -- depends on kernel on object_type::input_type
		double radius;	// == distance( centroid, object.back )
	
		// sorted on distance to the centroid!
		// vector actually is a good model for this storage: becuase it is sorted, a very fast split will 
		// be possible. 
		typedef typename std::vector<key_type>::iterator object_iter;
		typedef typename std::vector<key_type> object_container;
		std::vector<key_type> object;
	};
	
	
	// internal tree representation
	// TODO ajust the adjacency_list ... 
	typedef boost::adjacency_list< boost::vecS, boost::vecS, 
	                               boost::directedS, node > graph_type;
	typedef typename boost::graph_traits< graph_type >::vertex_iterator vertex_iter;
	typedef typename boost::graph_traits< graph_type >::out_edge_iterator out_edge_iter;
	typedef typename boost::graph_traits< graph_type >::vertex_descriptor vertex_descriptor;
	
	
	struct closer_by: std::unary_function< key_type const&, bool > {
		
		closer_by( key_type const &c1, key_type const &c2, PropertyMap *p_map ): centroid_1(c1), centroid_2(c2), map(p_map) {}
		bool operator()( key_type const &key ) {
			return distance_square( (*map)[key], (*map)[ centroid_1 ] ) < 
			       distance_square( (*map)[key], (*map)[ centroid_2 ] );
		}
		key_type centroid_1, centroid_2;
	 	PropertyMap *map;
	};
	
	
	
	//typedef tree<node<Object> > tree_type;
	//typedef typename tree<node<Object> >::iterator tree_iterator_type;

	m_tree( map_type map ): pattern_map(map) {
		
		// keep a reference to the property map
		std::cout << "m_tree constructor called..." << std::endl;
		std::cout << "Creating the (first) root node ... " << std::endl;
		root = boost::add_vertex( m_graph );
	}

	
 	void insert( key_type const &key ) {
 		
		std::cout << "inserting key " << key << " into the M-tree... " << std::endl;
		std::cout << "data at key " << key << " is " << pattern_map[ key ] << std::endl;

		// should look for the most appropriate node to put this point in 
		std::cout << "Looking for possible node to insert this object..." << std::endl;
		
		vertex_descriptor result = single_way( root, key );
	
		// suppose we get returned the new vertex... 
		if ( result != root ) {
			std::cout << "creating a new root node..." << std::endl;
			vertex_descriptor old_root = root;
			root = boost::add_vertex( m_graph );
			std::cout << "creating edge " << root << "->" << old_root << std::endl;
			std::cout << "creating edge " << root << "->" << result << std::endl;
			boost::add_edge( root, old_root, m_graph );
			boost::add_edge( root, result, m_graph );
		}

		print_vertices();
 	}

	
	// greedy search to nearest centroid: to a depth-first search to the centroids closest by


	// see Revisiting M-tree ... 



	/*!
	This is the original way of inserting an object into the M-tree. 
	- TODO Minimise the increase in covering radius
	- DONE Choose the centroid clostest by, and descend the tree (greedy search)

	in M-tree momenclature, routing object equals centroid of the child!!
	but I don't want a vector of centroids of the childs.
        I store a radius one level lower than in the original paper for the same reasons

	no bookkeeping of the radius until the tree has more than 1 node?
        root node does not have a centroid!

	* use the BGL properly: use a visitor class
	* first, a down pass (single way), to find the leaf node where to put the object in	
		 (use a predecessor recorder)
	* then, an upward pass (using the predecessor recordings) to update the tree.

	* of course, i.e., the centroid of a node _is_ the node's location (was the same in the recursive gonzales)
	* so indeed: a inner node has a centroid: but that centroid is a centroid of a node.

	* all objects are in the ground level.
	* n-ary tree? ... 

	*/
	// see section 3.3, at the insert pseudo-code

	vertex_descriptor single_way( vertex_descriptor const &node, key_type const &key ) {
		// if N is not a leaf then
		if ( boost::out_degree( node, m_graph ) > 0 ) {
			// do we have child nodes _in which_ this node could reside?

			// NO, we do not have child nodes _in which_ this node could reside.
			// Choose the one with minimum radius
			// Adjust radius of child node
			
			// there are child nodes, visit these if possible

			double min_dist = 1e99;
			vertex_descriptor min_node;

			typedef typename boost::graph_traits< graph_type >::adjacency_iterator adj_iter;
			for( std::pair<adj_iter,adj_iter> child = boost::adjacent_vertices(node, m_graph); child.first != child.second; ++child.first ) {
				std::cout << "processing a child node... " << std::endl;
				double dist2node = distance_square( pattern_map[key], pattern_map[ m_graph[ *child.first ].centroid ] );
				std::cout << "distance: " << dist2node << std::endl;
				if ( dist2node < min_dist) {
					min_dist = dist2node;
					min_node = *child.first;
				}
			}

			vertex_descriptor result = single_way( min_node, key );
			std::cout << "returned from recursive call.." << std::endl;

			if ( result == min_node )
				// nothing happened
				return node;
			else {
				// tie this new node to the current node. (no inner node splits yet)
				// overflow on node level right here!
				std::cout << "creating edge " << node << "->" << result << std::endl;
				boost::add_edge( node, result, m_graph );
				
				if( boost::out_degree( node, m_graph ) > 2 ) {
					split_inner_node( node );
				}
				
				return node;
			}
		} else {
			std::cout << "At a leaf node, inserting object..." << std::endl;
			m_graph[ node ].object.push_back( key );
			std::cout << "# of keys in node: " << m_graph[ node ].object.size() << std::endl;

			if ( m_graph[ node ].object.size() > 5 )
				return split_ground_node( node );
			else
				return node;
		}
	}




	vertex_descriptor nearest_centroid( vertex_descriptor const &node, key_type const &key, double &min_dist ) {

		std::cout << "dist2centroid " << distance_square( pattern_map[key], pattern_map[ m_graph[ node ].centroid ] ) << std::endl;
		std::cout << "min_dist: " << min_dist << std::endl;

		if ( boost::out_degree( node, m_graph ) > 0 ) {
			// there are child nodes, visit these if possible
			typedef typename boost::graph_traits< graph_type >::adjacency_iterator adj_iter;
			for( std::pair<adj_iter,adj_iter> child = boost::adjacent_vertices(node, m_graph); child.first != child.second; ++child.first ) {
				std::cout << "processing a child node... " << std::endl;
				// check a few conditions...

				// is this node within the radius of this node, or doesn't that matter?
				// no, should not matter.

				// expected minimum distance can be computed! ... ( dsitance - radius )

				std::cout << "visiting " << *child.first << std::endl;
				nearest_centroid( *child.first, key, min_dist );
			}
		}

		// else {

		double dist = distance_square( pattern_map[key], pattern_map[ m_graph[ node ].centroid ] );
		return node;
	
		//}
	}


	void add_to_node( vertex_descriptor const &node, key_type const &key ) {
	}


	// the split procedure, works bottom-up, starting at node
	vertex_descriptor split_ground_node( vertex_descriptor const &node ) {

		// ---> the centroid of "node", if it had one, will be ignored

		// alright, a new node will be added to the tree.
		// split the current node 

		std::cout << "split algorithm called. " << std::endl;

		// the object which has the _largest_ distance to the centroid will become the new centroid of the new node.

		// enhancements later on... now, just pick the last point and promote that to a node
		vertex_descriptor new_node = boost::add_vertex( m_graph );
		std::cout << "created a new vertex..." << new_node << std::endl;


		// perform 2-center...
		// should be an algorithm chosen for simplicity.

		// pick a "random" 1st center.

		typename node::object_iter i( m_graph[node].object.begin() );
		key_type centroid_1 = *i++;
		key_type centroid_2 = *i++;
		
		two_center( centroid_1, centroid_2 );
		

		double max_dist = distance_square( pattern_map[centroid_1], pattern_map[centroid_2] );

		
		// can also be 2 aribtrarily random points!
		// search for the point furthest away
		for( ; i != m_graph[node].object.end(); ++i ) {
			double dist = distance_square( pattern_map[centroid_1], pattern_map[ *i ] );
			std::cout << "distance to point: " << dist << std::endl;
			if ( dist > max_dist ) {
				max_dist = dist;
				centroid_2 = *i;
			}
		}

		// TODO
		// try to iteratively improve these found centroids ... 
		// TODO

		// right now, it is 0 1 2 centroid 0, but centroid 1 and 4 are better
                //                  3 4 5 centroid 5 

		// partition the objects based on these centroids (rather efficient!)
		// first..middle belongs to centroid_1, and middle..end belongs to centroid 2

		//  TODO use a functor to store the radii of the nodes.
		typename node::object_iter middle = std::partition(  m_graph[node].object.begin(), m_graph[node].object.end(), 
								     closer_by(centroid_1,centroid_2,pattern_map) );
                                 //boost::bind( &this_type::closer_by,boost::ref(*this),centroid_1,centroid_2,_1) );

		// move the appropriate objects to the other node		
		std::copy( middle, m_graph[node].object.end(), std::back_inserter( m_graph[new_node].object ) );
		m_graph[node].object.erase( middle, m_graph[node].object.end() );

		std::cout << "Centroid 1 " << centroid_1 << std::endl;
		std::cout << "Centroid 2 " << centroid_2 << std::endl;

		// assign centroids
		m_graph[ node ].centroid = centroid_1;
		m_graph[ new_node ].centroid = centroid_2;

		return new_node;
	}


	
	
	void two_center( key_type const &begin, key_type const &end ) {
	
		// TODO put in tree global memory
		// for splits at 6
		boost::uniform_int<> int_dist(0,5);
		boost::variate_generator< boost::mt19937&,boost::uniform_int<> > node_picker(rng,int_dist);
	
		// pick a center at random
		
		unsigned int nr = node_picker();
		
		std::cout << "Picked centroid 1 to be " << nr << std::endl;
		
		
		//std::partition( begin, end, closer_by( ));
	
	}
	
	
	void split_inner_node( vertex_descriptor const &node ) {

		std::cout << "Split inner node called.." << std::endl;	
	
		// so, this is about the out-edges...
	
		vertex_descriptor new_node = boost::add_vertex( m_graph );
		std::cout << "created a new vertex..." << new_node << std::endl;

		// just pick 2 edges ... to be improved
		
		std::pair<out_edge_iter,out_edge_iter> edges = boost::out_edges( node, m_graph );
		
		key_type centroid_1 = m_graph[boost::target(*edges.first,m_graph)].centroid;
		key_type centroid_2 = m_graph[boost::target(*(edges.second-1),m_graph)].centroid;
		
		std::cout << "centroid 1 is " << centroid_1 << std::endl;
		std::cout << "centroid 2 is " << centroid_2 << std::endl;
		
		// alright, now split the nodes to these centroids
		//out_edge_iter middle = std::partition( edges.first, edges.second, 
		//		                       closer_by(centroid_1,centroid_2,pattern_map) );
		
		
			
	}







	void erase( key_type const &key ) {
		std::cout << "removing key " << key << " from the m-tree..." << std::endl;
	}
	
	
	void print_vertices() {

		// TODO use a DFS visitor		
		std::cout << std::endl << "Tree configuration:" << std::endl << std::endl;
	
		std::cout << " root: " << root << std::endl;

		std::pair<vertex_iter, vertex_iter> vp;
		for( vp = boost::vertices(m_graph); vp.first != vp.second; ++vp.first ) {
			std::cout << "-- id: " << *vp.first << std::endl;
			std::cout << "          centroid: " << m_graph[*vp.first].centroid << std::endl;
			std::cout << "          radius:   " << m_graph[*vp.first].radius << std::endl;
			std::cout << "          objects:  ";
			std::copy( m_graph[*vp.first].object.begin(), m_graph[*vp.first].object.end(), std::ostream_iterator<key_type>(std::cout, " ") );
			std::cout << std::endl;
		}
	
	
		boost::print_graph( m_graph );
	
	}


	void slim_down() {

		// perhaps local_iterative_improvement is a better name. Or something like that.

		// this is equal to the algorithm for k-means, but then applied for 
		// k-center. However, an improvement is only an improvement if _BOTH_
		// radii decrease (i.e. not min_max). 
	

	}
	

	// The (tree) graph structure, and the iterator of the root node.
	graph_type m_graph;
	vertex_descriptor root;

	// pattern_map
	PropertyMap pattern_map;

	// a random number generator
	boost::mt19937 rng; 
	

};






} //namespace kml

#endif

