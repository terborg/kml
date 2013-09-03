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

#ifndef IO_HPP
#define IO_HPP

#include <boost/lexical_cast.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/size.hpp>
#include <boost/range/value_type.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/tuple/tuple.hpp>

#include <kml/input_value.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

#include <boost/range.hpp>

// temp
#include <boost/numeric/ublas/io.hpp>



namespace kml {


namespace io {
enum problem_type { unknown, regression, classification, ranking };



/*!
 
The .dst file format is very simple: comma delimited lines of text.
 
Lines that start with "A" describe a variable, whose id number is the
second field in the line. The third field in the line is either 101 (if the
variable is a class label) or 1 (if the variable is not). For this
benchmark, all you need to use is that variable id 1000 is the class
label, while variable ids from 1001 to 1123 are input attributes.
 
Lines that start with "C" indicate the start of a new training example. The
rest of the line identifies the training example.
 
Lines that start with "V" are attributes of the current example. The second
field is the variable id (see "A" lines above), and the third field is the
value of the variable. If a variable is not specified by a "V" line, its
value defaults to 0. 
 
For example, in this benchmark, if an example has a line that says 
 
V,1000,0
 
it means that the example has a negative label. If the following line
appears:
 
V,1015,1
 
it means that input attribute #15 is true (1). If no such line appears
before the next "C" line, it means that input attribute #15 is false (0).
 
*/
namespace dst {

bool compatible( std::vector<std::string> const &container ) {
    return ( (container[0][0]=='A') || (container[0][0]=='V') || (container[0][0]=='C') );
}

problem_type problem_type( std::vector<std::string> const &container ) {
    return classification;
}

template<typename PropertyMap, typename BackInsertionSequence>
void read( std::vector<std::string> const &container, io::problem_type p_type,
           PropertyMap &map, BackInsertionSequence &keys ) {

    typedef PropertyMap map_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    // should become some KML shortcut to do this...
    // should do a run-time specialisation per problem_type, as this is probably problem_type specific
    typedef typename boost::tuples::element<0,object_type>::type input_type;
    typedef typename boost::tuples::element<1,object_type>::type output_type;

    std::map<unsigned int, int> id_mapping;
    unsigned int output_class_id(0);
    bool output_class_found(false);

    // first read format description
    std::vector<std::string>::const_iterator i = container.begin();
    boost::char_separator<char> csv_separator(",\n\r");
    unsigned int max_attributes = 0;
    while( (i!=container.end()) && ((*i)[0]=='A') ) {
        boost::tokenizer<boost::char_separator<char> > values( *i, csv_separator );
        boost::tokenizer<boost::char_separator<char> >::iterator j = values.begin();

        // skip past the A
        ++j;

        // read a possible class id
        unsigned int id = boost::lexical_cast< unsigned int >( *j++ );

        if ( boost::lexical_cast< unsigned int >( *j ) == 101 ) {
            output_class_id = id;
            output_class_found = true;
        } else {
            id_mapping[ id ] = max_attributes++;
        }

        ++i;
    }

    if (!output_class_found) {
        std::cout << "Serious problem, could not find an output class id." << std::endl;
    }

    int example_counter(0);
    int current_example(0);

    output_type output(0);

    std::cout << "max attributes: " << max_attributes << std::endl;
    input_type attributes( max_attributes );

    while(i!=container.end()) {
        boost::tokenizer<boost::char_separator<char> > tokens( *i, csv_separator );
        boost::tokenizer<boost::char_separator<char> >::iterator j = tokens.begin();

        switch( (*i)[0] ) {
            // the start of a new training example
        case 'C': {
                if ( current_example != example_counter ) {
                    map[ current_example ] = boost::make_tuple( attributes, output );
                    keys.push_back( current_example );
                }
                //std::cout << "new example..." << std::endl;
                current_example = example_counter;
                attributes.clear();
                ++example_counter;
                break;
            }
            // attributes of the new training example
        case 'V': {
                // detect whether it is the output class type, or not...
                ++j; // skip the 'V'

                unsigned int id = boost::lexical_cast< unsigned int >( *j++ );
                if ( id == output_class_id ) {
                    if (boost::is_same<output_type, bool>::value)
                        output = (boost::lexical_cast<int>(*j) > 0);
                    else
                        output = boost::lexical_cast<output_type>(*j);
                    // set the output to something
                }
                else {
                    attributes[ id_mapping[ id ] ] = boost::lexical_cast< double >( *j );
                }
                break;
            }
        default: {
                //std::cout << "Error detected in input file." << std::endl;
                break;
            }
        }
        ++i;
    }
}

}



/*!
Read a data file also used in Joachims' SVM^{light} format
For more details, see http://svmlight.joachims.org/
 
 
\param I a range over input examples
\param O a range over output examples
 
The routine performs two passes over the data file (mainly to 
preserve the maximum amount of memory needed): 
 
-1 Acquire statistics of the file
 
   - How many data samples are we dealing with?
   - What is the maximum number of nonzero attributes found in a sample?
   - What is the maximum index of attribute found in a sample?
 
-2 Read and parse all values in the file
 
\ingroup fileio
 
 
The input file example_file contains the training examples. 
The first lines may contain comments and are ignored if they start with #. 
Each of the following lines represents one training example and is of the following format: 
 
<line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
<target> .=. +1 | -1 | 0 | <float> 
<feature> .=. <integer> | "qid"
<value> .=. <float>
<info> .=. <string> 
 
The target value and each of the feature/value pairs are separated by a space character. 
Feature/value pairs MUST be ordered by increasing feature number. 
Features with value zero can be skipped. The string <info> can be used to pass additional 
information to the kernel (e.g. non feature vector data).
 
 
*/




namespace svm_light {

bool compatible( std::vector<std::string> const &container ) {
    std::vector<std::string>::const_iterator i = container.begin();

    // check for <line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value>
    boost::char_separator<char> separator(" \t");
    boost::tokenizer<boost::char_separator<char> > first_line( *i, separator );
    boost::tokenizer<boost::char_separator<char> >::iterator j = first_line.begin();

    // non-empty line is no good
    if ( j == first_line.end() )
        return false;
    // skip the target
    ++j;
    // no feature data?
    if ( j == first_line.end() )
        return false;

    // perform the following test: each token should contain a ':'
    bool format_ok = true;
    while( (j != first_line.end()) && format_ok ) {
        format_ok = std::find( boost::begin(*j), boost::end(*j), ':' ) != boost::end(*j);
        ++j;
    }

    return format_ok;
}


problem_type problem_type( std::vector<std::string> const &container ) {

    // try to reject binary classification; each line should start with +1 or with -1
    std::vector<std::string>::const_iterator i = container.begin();
    while( (i!=container.end()) && ((*i).size()>3) && ((*i)[1]=='1') && ( ((*i)[0]=='+') || ((*i)[0]=='-') ) )
        ++i;
    if (i==container.end())
        return io::classification;

    // try to reject ranking; each line should contain 'qid'
    i = container.begin();
    bool found_qid = true;
    while ( i!=container.end() && found_qid ) {
        std::string::const_iterator q_loc = std::find( (*i).begin(), (*i).end(), 'q' );
        found_qid = (((*i).end()-q_loc)>2) && (q_loc[1]=='i') && (q_loc[2]=='d');
        ++i;
    }
    if (found_qid)
        return io::ranking;

    // no other tests available, default to regression
    return regression;
}


template<typename PropertyMap, typename BackInsertionSequence>
void read( std::vector<std::string> const &container, io::problem_type p_type,
           PropertyMap &map, BackInsertionSequence &keys ) {

    typedef PropertyMap map_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;
    typedef typename boost::tuples::element<0,object_type>::type input_type;
    typedef typename boost::tuples::element<1,object_type>::type output_type;

    // quick routine to figure out the maximum feature number
    std::vector<std::string>::const_iterator iter = container.begin();
    std::size_t max_attribute_nr = 0;

    // because the feature values are ordened by design, we only need the LAST feature value
    while( iter != container.end() ) {
        // start at the end of the line, backwards search for the first occurance of ":",
        // find the atrribute number of value in front of that
        typedef boost::range_const_reverse_iterator<boost::tokenizer<boost::char_separator<char> >::value_type>::type char_iter_type;
        char_iter_type char_iter = std::find( boost::const_rbegin(*iter), boost::const_rend(*iter), ':' );
        ++char_iter;
        unsigned int attribute_nr = 0;
        unsigned int multiplier = 1;
        while( (*char_iter) != ' ' ) {
            attribute_nr += boost::lexical_cast< unsigned int >( *char_iter ) * multiplier;
            multiplier *= 10;
            ++char_iter;
        }
        if (attribute_nr > max_attribute_nr)
            max_attribute_nr = attribute_nr;
        ++iter;
    }

    //std::cout << "max attribute number: " << max_attribute_nr << std::endl;

    // these are the separators the input file will be split on:
    // space, tab, #, and :
    boost::char_separator<char> separator(" \t#:");

    // reset the data iterator
    iter = container.begin();
    unsigned int sample_key = 0;
    while (iter != container.end()) {

        boost::tokenizer<boost::char_separator<char> > tokens( *iter, separator );
        boost::tokenizer<boost::char_separator<char> >::iterator attribute_iter = tokens.begin();

        output_type output(0);
        if (boost::is_same<output_type, bool>::value)
            output = (boost::lexical_cast<int>(*attribute_iter++) > 0);
        else
            output = boost::lexical_cast< output_type >( *attribute_iter++ );

	// create the input type, which is in this case (still)
	// assumed to be a dense vector (e.g., std::vector, or ublas::vector). 
	// clear the input type, it could be uninitialised
        input_type attributes( max_attribute_nr );
	std::fill( attributes.begin(), attributes.end(), 0.0 );

        while( (attribute_iter != tokens.end()) && ((*iter)[0] != static_cast<char>('#')) ) {

            // FIXME
            // ERROR: this is not in all cases a number, such as in case of a ranking problem
            // Figure out the attribute number. This is 1-based, so subtract 1
            unsigned int attribute_nr = boost::lexical_cast< unsigned int >( *attribute_iter++ ) - 1;
            // read the input into the input container
            attributes[ attribute_nr ] = boost::lexical_cast< double >( *attribute_iter++ );
        }

        map[ sample_key ] = boost::make_tuple( attributes, output );
        keys.push_back( sample_key );

        ++sample_key;
        ++iter;
    }
}
}



/*!
Read a data file also used in SVM Torch - train data file format in ASCII mode
For more details, see http://www.idiap.ch/machine_learning.php?content=Torch/en_SVMTorch.txt
 
\param I a range over input examples
\param O a range over output examples
 
Because you supply inputs AND outputs here, the SVM Torch Train data file format is assumed
 
\ingroup fileio
*/

namespace svm_torch {



bool compatible( std::vector<std::string> const &container ) {
    boost::char_separator<char> separator(" \t");
    std::vector<std::string>::const_iterator i = container.begin();
    try {
        boost::tokenizer<boost::char_separator<char> > first_line( *i, separator );
        boost::tokenizer<boost::char_separator<char> >::iterator j = first_line.begin();
        if ( j == first_line.end() )
            return false;
	//        unsigned int samples = boost::lexical_cast<unsigned int>( *j++ );
	//        unsigned int attributes = boost::lexical_cast<unsigned int>( *j++ );
    } catch( boost::bad_lexical_cast & ) {
        return false;
    }

    return true;
}


io::problem_type problem_type( std::vector<std::string> const &container ) {
    boost::char_separator<char> separator(" \t");
    // try to detect the problem type

    // skip the first line
    std::vector<std::string>::const_iterator i = container.begin();
    ++i;

    bool classification_assumption = true;

    // try to reject binary classification
    while( classification_assumption && (i!=container.end()) ) {
        std::string::const_iterator j( (*i).end() );
        // first, find a non-space character
        bool found = false;
        while( !found && (j != (*i).begin()) ) {
            --j;
            found = (*j != ' ');
        }
        std::string::const_iterator k( j );
        ++k;
        // then, find the first space character
        found = false;
        while( !found && (j != (*i).begin()) ) {
            --j;
            found = (*j == ' ');
        }
        if (found)
            ++j;
        double value = boost::lexical_cast< double >( std::string(j,k) );
        classification_assumption = ( (value==1.0) || (value==-1.0) );
        ++i;
    }

    if (classification_assumption)
        return io::classification;

    // no other tests yet, return regression
    return io::regression;
}



template<typename PropertyMap, typename BackInsertionSequence>
void read( std::vector<std::string> const &container, io::problem_type p_type,
           PropertyMap &map, BackInsertionSequence &keys ) {

    typedef PropertyMap map_type;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;

    // should become some KML shortcut to do this...
    typedef typename boost::tuples::element<0,object_type>::type input_type;
    typedef typename boost::tuples::element<1,object_type>::type output_type;

    unsigned int number_of_samples;
    unsigned int number_of_attributes;

    // these are the separators the input file will be split on:
    // space and tab
    boost::char_separator<char> separator(" \t");

    std::vector<std::string>::const_iterator iter = container.begin();

    boost::tokenizer<boost::char_separator<char> > first_line( *iter, separator );
    boost::tokenizer<boost::char_separator<char> >::iterator i=first_line.begin();
    number_of_samples = boost::lexical_cast< unsigned int >( *i++ );
    number_of_attributes = boost::lexical_cast< unsigned int >( *i ) - 1;

    std::cout << "nr of samples: " << number_of_samples << std::endl;
    std::cout << "nr of attributes: " << number_of_attributes << std::endl;

    // go to the next line...
    ++iter;

    // read the data
    unsigned int sample_index = 0;
    input_type my_attributes( number_of_attributes );
    output_type output;

    while( iter != container.end() ) {
        // split the current line
        boost::tokenizer<boost::char_separator<char> > tokens( *iter, separator );
        i = tokens.begin();

        // DENSE vector format
        for( unsigned int j=0; j<number_of_attributes; ++j ) {
            my_attributes[j] = boost::lexical_cast< double >( *i++ );
        }


        if (boost::is_same<output_type, bool>::value)
            output = (boost::lexical_cast<double>(*i) > 0.0);
        else
            output = boost::lexical_cast< output_type >( *i );

        map[ sample_index ] = boost::make_tuple( my_attributes, output );
        keys.push_back( sample_index );

        ++sample_index;
        ++iter;
    }
}

} // namespace svm_torch




// basic filetype: a data matrix, i.e., one example per line

namespace data_matrix {


//
bool compatible( std::vector<std::string> const &container ) {
    return true;
}



io::problem_type problem_type( std::vector<std::string> const &container ) {
    return io::regression;
}






template<typename TokenIterator, std::size_t TupleSize>
struct line_reader {};

template<typename TokenIterator>
struct line_reader<TokenIterator,0> {
	line_reader() {
		std::cout << "Not a tuple type..." << std::endl;	
	}
};

template<typename TokenIterator>
struct line_reader<TokenIterator,1> {
	line_reader() {
		std::cout << "Tuple size = 1, instantiated." << std::endl;	
	}
};




template<typename PropertyMap, typename BackInsertionSequence>
void read( std::vector<std::string> const &container, io::problem_type p_type,
           PropertyMap &map, BackInsertionSequence &keys ) {

    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef typename boost::property_traits<PropertyMap>::value_type object_type;
    typedef typename boost::tuples::element<0,object_type>::type input_type;
    typedef typename boost::tuples::element<1,object_type>::type output_type;

    
//     std::cout << "starting to read the data matrix filetype..." << std::endl;

    // separate on a superset of a csv separator
    boost::char_separator<char> separator(", \t\n\r");
    std::vector<std::string>::const_iterator i = container.begin();
    
    
    // check properties of first line
    boost::tokenizer<boost::char_separator<char> > first_line( *i, separator );
    boost::tokenizer<boost::char_separator<char> >::iterator line_tok = first_line.begin();
    unsigned int nr_of_attributes = 0;
    std::vector<bool> process_column;
    while( line_tok != first_line.end() ) {
	    ++line_tok;
	    ++nr_of_attributes;
    }
    
    //std::cout << "nr of columns: " << nr_of_attributes << std::endl;
    
    input_type row( nr_of_attributes );
    key_type sample_index = 0;
    
    while( i != container.end() ) {
        boost::tokenizer<boost::char_separator<char> > values( *i, separator );
        boost::tokenizer<boost::char_separator<char> >::iterator j = values.begin();

        int row_entry = 0;
        while( j != values.end() ) {
			try {	        
	        	row[row_entry++] = boost::lexical_cast<double>(*j);
    	    } catch ( boost::bad_lexical_cast & ) {
	    	    row[row_entry++] = std::numeric_limits<double>::quiet_NaN();
       	    }
	        ++j;
        }
        
	map[ sample_index ] = boost::make_tuple( row, boost::lexical_cast<output_type>(0.0) );
       keys.push_back( sample_index );
       ++sample_index;
        ++i;
    }


}

} // namespace data_matrix




} // namespace kml::io






class file {
public:
    typedef boost::char_separator<char> char_separator_type;
    typedef boost::tokenizer< char_separator_type, char* > char_tokenizer;


    file( std::string const &filename ) {

        // open the file
        std::ifstream input_file( filename.c_str(), std::ios::in );
        if ( !input_file.is_open() ) {
            std::cout << "Could not open file " << filename << std::endl;
        }
        std::cout << "Reading " << filename << "..." << std::flush;
        std::string line_buffer;
        while( !input_file.eof() ) {
            std::getline( input_file, line_buffer );
            // skip empty lines and lines that begin with a comment mark
            if ((line_buffer.size()>0) && (line_buffer[0]!='#')) {
                // strip comments (i.e., anything after '#') from these lines
                std::string::iterator my_end = std::find( line_buffer.begin(), line_buffer.end(), '#' );
                buffer.push_back( std::string(line_buffer.begin(), my_end) );
            }
        }
        std::cout << "read " << buffer.size() << " lines." << std::endl;
        input_file.close();

        if ( buffer.size() == 0 ) {
            std::cout << "Could not read any contents from " << filename << std::endl;
        }

        if ( io::dst::compatible( buffer) ) {
            handler = dst_handler;
            //std::cout << "dst handler installed" << std::endl;
        } else
            if ( io::svm_light::compatible( buffer ) ) {
                handler = svm_light_handler;
                //std::cout << "svm light handler installed" << std::endl;
            } else
                if ( io::svm_torch::compatible( buffer ) ) {
                    handler = svm_torch_handler;
                    //std::cout << "svm torch handler installed" << std::endl;
                } else
                    if ( io::data_matrix::compatible( buffer ) ) {
                        handler = data_matrix_handler;
                        //std::cout << "data matrix handler installed" << std::endl;
                    } else
                        std::cout << "unknown file format!" << std::endl;

    }

    io::problem_type problem_type() {
        switch( handler ) {
        case dst_handler: {
                return io::dst::problem_type( buffer );
                break;
            }
        case svm_light_handler: {
                return io::svm_light::problem_type( buffer );
                break;
            }
        case svm_torch_handler: {
                return io::svm_torch::problem_type( buffer );
                break;
            }
        case data_matrix_handler: {
                return io::data_matrix::problem_type( buffer );
                break;
            }
        }
        return io::unknown;
    }

    template<typename PropertyMap, typename BackInsertionSequence>
    void read( PropertyMap &map, BackInsertionSequence &keys ) {
        switch( handler ) {
        case dst_handler: {
                io::dst::read( buffer, io::classification, map, keys );
                break;
            }
        case svm_light_handler: {
                io::svm_light::read( buffer, io::classification, map, keys );
                break;
            }
        case svm_torch_handler: {
                io::svm_torch::read( buffer, io::classification, map, keys );
                break;
            }
        case data_matrix_handler: {
                io::data_matrix::read( buffer, io::classification, map, keys );
                break;
            }
        }
    }

    enum file_handler { dst_handler, svm_light_handler, svm_torch_handler, data_matrix_handler };

    file_handler handler;

    std::vector<std::string> buffer;
};





// toy example of using bzip2 compression

/*
template<typename PatternMap>
void write( char* file_name, PatternMap &dataset ) {
 
    // open the file
    std::ofstream output_file( file_name, std::ios::out | std::ios::binary );
 
    // define the output filter
    iostreams::filtering_ostream out;
    out.push( iostreams::bzip2_compressor() );
    out.push( output_file );
    
    // perform a test
    out << "Test!" << std::endl;
    
    // close everything
    out.pop();
    output_file.close();
    
}
*/




} // namespace kml



#endif

