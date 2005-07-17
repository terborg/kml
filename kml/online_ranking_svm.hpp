/***************************************************************************
*  online_ranking_svm.hpp: Copyright (C) 2005 by Meredith L. Patterson     *
*  intended for use with                                                   *
*  The Kernel-Machine Library                                              *
*  Copyright (C) 2004, 2005 by Rutger W. ter Borg                          *
*                                                                          *
*  This program is free software; you can redistribute it and/or           *
*  modify it under the terms of the GNU General Public License             *
*  as published by the Free Software Foundation; either version 2          *
*  of the License, or (at your option) any later version.                  *
*                                                                          *
*  This program is distributed in the hope that it will be useful,         *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
*  GNU General Public License for more details.                            *
*                                                                          *
*  You should have received a copy of the GNU General Public License       *
*  along with this program; if not, write to the Free Software             *
*  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307   *
***************************************************************************/

#include <algorithm>
#include <functional>

#include "online_svm.hpp"

namespace ublas = boost::numeric::ublas;

namespace kml {

  template<typename Input, typename Rank, template<typename, int> class Kernel>
  class online_ranking_svm: public online_svm<Input, Rank, Kernel> {
  public:
    typedef online_svm<Input,Rank,Kernel> base_type;
    typedef typename base_type::kernel_type kernel_type;
    typedef typename base_type::result_type result_type;
    typedef Input input_type;
    typedef Rank rank_type;
    typedef typename base_type::scalar_type scalar_type;
    typedef ublas::symmetric_matrix<double> symmetric_type;
    typedef ublas::matrix<double> matrix_type;
    typedef ublas::vector<double> vector_type;
    typedef typename boost::range_value<vector_type>::type point_type;

    online_ranking_svm(typename boost::call_traits<kernel_type>::param_type k,
		       typename boost::call_traits<double>::param_type param_eps,
		       typename boost::call_traits<double>::param_type param_C): 
	base_type(k, param_eps, param_C) { }

    void learn(vector_type const &input, rank_type const &rank) {
      for (vector_type::iterator i = input.begin(); i != input.end(); ++i) {
	for (vector_type::iterator j = i+1; j != input.end(); ++j) {
	  if (rank[i] != rank[j]) {
	    input_type d;
	    std::transform(i->begin(), 	i->end(), j->begin(), d.begin(), std::minus<point_type>());
	    pairwise.push_back(d);
	    classes.push_back(rank[i] > rank[j]);
	  }
	}
      }
      kml::online_svm<Input, bool, Kernel>::learn(pairwise, classes);
    }
    ublas::vector<input_type> pairwise;
    ublas::vector<bool> classes;      
  };
}

    
