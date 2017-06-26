#if !defined(NLTK_CACHE_H)
#define NLTK_CACHE_H

#include <atomic>
#include <algorithm>
#include "tbb/parallel_for_each.h"
#include "offsetstores.h"
#include <vector>
#include <string>
#include <mutex>
#include <iterator>
#include <iomanip>
#include <thread>

#include "wnb/core/wordnet.hh"
#include "wnb/nltk_similarity.hh"

using namespace std;

class nltk_cache
{
public:
	//wnb::wordnet &wn;
	//wnb::nltk_similarity nltkSim;

	const std::string similaritiesFilename = "similarities";
	offset::OffsetMatrix<uint8_t> wnSimilarities;

//		uint8_t u = f1 < 0.f ? 255 : (f1*254.0f)+0.5f;
//		float f2 = u == 255 ? -1.f : (1.f / 254.f) * u;

	static uint8_t f_to_b( float f )
	{
		return round( pow( f, -1.f ) );
		//return f == -1 ? 0 : uint8_t((f*255.f)+0.5f);
	}

	static float b_to_f( uint8_t b )
	{
		return b == 255 ? -1.f : pow( b, -1.f );
		//return (1.f / 255.f) * b;
	}

	float f_lookup[256];

	float b_to_f_cached( uint8_t b ) const
	{
		return f_lookup[b];
	}

	void set_similarity( int a, int b, float sim )
	{
		wnSimilarities.set( a, b, f_to_b(sim) );
	}

public:
	bool empty() const { return wnSimilarities.empty(); }
	size_t count( const uint8_t val ) const { return wnSimilarities.count(val); }

	struct Row
	{
		size_t offset, from, size;
	};
	vector<Row> matrix;
	vector<uint8_t> values;

	void calculate_matrix( wnb::wordnet& wn, const bool verbose=false )
	{
		std::pair<boost::adjacency_list<>::vertex_iterator,
			boost::adjacency_list<>::vertex_iterator> vs = boost::vertices(wn.wordnet_graph);

		std::vector<wnb::synset> synsets;	// should really figure out how to do this without copying all to a vector first
		for( auto it=vs.first; it!=vs.second; ++it )
			synsets.emplace_back( wn.wordnet_graph[*it] );

		//synsets.resize(10); // DEBUG

		wnb::nltk_similarity nltkSim( wn );

		if( verbose ) std::cout << "Generate ";

		struct TempRow
		{	
			std::vector<uint8_t> data;
			size_t from;
		};

		std::vector<TempRow> tempmatrix( synsets.size() );

		atomic<size_t> progress(0);

		auto genfunc = [&]( const tbb::blocked_range<size_t> &range )
		{
			std::vector<float> buffer;

			for( size_t i=range.begin(); i<range.end(); ++i )
			{
				size_t sims = 0;

				const wnb::synset &a = *(synsets.begin()+i);
				TempRow &row = tempmatrix[a.id];

				buffer.resize( synsets.size()-a.id );
				for( auto b=synsets.begin()+a.id; b!=synsets.end(); ++b )
				{
					*(buffer.begin()+(b->id-a.id)) = nltk_cache::f_to_b(nltkSim(a,*b)); //nltkSim( a, *b );

					/*if( *(buffer.begin()+(b->id-a.id)) > 0.32 && *(buffer.begin()+(b->id-a.id)) < 0.34 )
						*(buffer.begin()+(b->id-a.id)) = -1;*/

					++sims;
				}

				auto begin = buffer.begin()+1;
				while( begin!=buffer.end() && *begin==-1 ) ++begin;

				auto end = buffer.end()-1;
				while( end>begin && *end==-1 ) --end;
				++end;

				row.data = vector<uint8_t>( begin, end );
				row.from = begin - buffer.begin() + a.id;

				progress += sims;
				cout << progress << "          \r" << flush;
			}
		};
		tbb::parallel_for( tbb::blocked_range<size_t>( 0, synsets.size() ), genfunc );
		//tbb::parallel_for_each( synsets.begin(), synsets.end(), genfunc );
	//	for_each( synsets.begin(), synsets.end(), func );


		cout << "calc offsets" << endl;

		matrix.resize( synsets.size() );
		if( !synsets.empty() ) matrix.front() = { 0, tempmatrix.front().from, tempmatrix.front().data.size() };

		for( size_t i=1; i<synsets.size(); ++i )
		{
			matrix[i] = { matrix[i-1].offset+matrix[i-1].size, tempmatrix[i].from, tempmatrix[i].data.size() };
		}

		cout << "create final matrix" << endl;

		const auto &b = matrix.back();
		values.resize( b.offset + b.size );

		auto copyfunc = [&]( const tbb::blocked_range<size_t> &range )
		{
			for( size_t i=range.begin(); i<range.end(); ++i )
			{
				const auto &temprow = tempmatrix[i];
				const auto &row = matrix[i];

				std::copy( temprow.data.begin(), temprow.data.end(), values.begin()+row.offset );
			}		
		};
		tbb::parallel_for( tbb::blocked_range<size_t>( 0, synsets.size() ), copyfunc );
		
		//std::cout << "matrix: " << tempmatrix.size() << std::endl;	

		//for( auto i : values ) cout << nltk_cache::b_to_f_cached( i ) << ", "; cout << endl;

		if( verbose ) std::cout << std::endl;
	}

	/* writes the currect Matrix as a binary file to filename.
		returns true if error, false if success.

		format is:
			[minimum row number]           [number of rows]
			 size_t                         size_t

			[minimum col number in 1st row][number of columns in 1st row]
			 size_t                         size_t
			[1st row, 1st col value][1st row, 2nd col value]...[1st row, last col value]
			 typedef T               typedef T                  typedef T

			[minimum col number in 2nd row][number of columns in 2nd row]
			 size_t                         size_t
			[2nd row, 1st col value][2nd row, 2nd col value]...[2nd row, last col value]
			 typedef T               typedef T                  typedef T

			...

			[minimum col number in last row][number of columns in last row]
			 size_t                         size_t
			[last row, 1st col value][last row, 2nd col value]...[last row, last col value]
			 typedef T                typedef T                   typedef T	
	*/

	bool save( const std::string& path, const bool verbose=true ) const
	{
		if( verbose) std::cout << "Save file" << std::endl;

		std::ofstream file( path+similaritiesFilename, std::ios::binary );
		if( !file.good() ) return true;

		// write the minimum row number and the number of rows
		const size_t total = values.size();
		const size_t rowsNum = matrix.size();
		//file.write( (char*)&total, sizeof(total) );
		file.write( (char*)&rowsNum, sizeof(rowsNum) );

		size_t progress = 0;

		// for each row in order
		for( const auto &row : matrix )
		{
			if( verbose )
			{
				std::cout << ++progress << "\r" << std::flush;
			}

			const size_t from = row.from;
			const size_t offset = row.offset;
			const size_t size = row.size;
			file.write( (char*)&from, sizeof(from) );
			file.write( (char*)&offset, sizeof(offset) );
			file.write( (char*)&size, sizeof(size) );

			file.write( (char*)values.data(), sizeof(uint8_t)*total );
		}

		if( verbose ) std::cout << std::endl;

		file.close();
		
		return false;


		//return wnSimilarities.save( path + similaritiesFilename );
	}


	/* get the similarity between two synsets as a byte*/
	/*uint8_t similarity_b( const wnb::synset &a, const wnb::synset &b )
	{
		if( wnSimilarities.empty() ) return f_to_b( similarity_on_fly(a,b) );

		return wnSimilarities.get( min(a.id,b.id), max(a.id,b.id) );
	}*/

	/* get the similarity between two synsets */
	/*float similarity( const wnb::synset &a, const wnb::synset &b ) 
	{
		if( wnSimilarities.empty() ) return similarity_on_fly(a,b);

		return b_to_f(wnSimilarities.get( min(a.id,b.id), max(a.id,b.id) ));
	}*/

	static std::vector<std::string> split( const std::string &text, char delimiter='|' )
	{
		std::vector<std::string> tokens;

	    size_t s, e;
	    for( s=0, e=text.find(delimiter);
			e!=std::string::npos;
			s=e+1, e=text.find(delimiter,s) )
		{
			tokens.emplace_back( text.substr( s, e-s) );
		}

		e = text.length()-s;
		if( e )
			tokens.emplace_back( text.substr( s, e ) );

		for( auto &tok : tokens )
			std::transform( std::begin(tok), std::end(tok), std::begin(tok), ::tolower );

		return tokens;
	}

	typedef std::vector< wnb::synset > Synsets;

	float operator()( const wnb::synset& a, const wnb::synset& b ) const
	{
		if( empty() ) return -1.f;

		if( a == b ) return 1.f;

		return b_to_f_cached( wnSimilarities.get( std::min(a.id,b.id), std::max(a.id,b.id) ) );
	}

	nltk_cache() : wnSimilarities( f_to_b(-1.f) )
	{
		for( int i=0; i<256; ++i )
		{
			f_lookup[i] = b_to_f( i );
		}
	}

	nltk_cache( const std::string& path, const bool verbose=false ) : nltk_cache()
	{
		wnSimilarities.load( path + similaritiesFilename, verbose );
	}

	nltk_cache( wnb::wordnet& wn, const bool verbose=false ) : nltk_cache()
	{
		calculate_matrix( wn, verbose );
	}
};

#endif