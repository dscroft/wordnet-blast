#if !defined(NLTK_CACHE_H)
#define NLTK_CACHE_H

#include <atomic>
#include <algorithm>
#include "tbb/parallel_for_each.h"
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
	static const uint8_t nullsim = 255;
	//offset::OffsetMatrix<uint8_t> wnSimilarities;

//		uint8_t u = f1 < 0.f ? 255 : (f1*254.0f)+0.5f;
//		float f2 = u == 255 ? -1.f : (1.f / 254.f) * u;

	static uint8_t f_to_b( float f )
	{
		return round( pow( f, -1.f ) );
		//return f == -1 ? 0 : uint8_t((f*255.f)+0.5f);
	}

	static float b_to_f( uint8_t b )
	{
		return b == nullsim ? -1.f : pow( b, -1.f );
		//return (1.f / 255.f) * b;
	}

	float f_lookup[256];

	inline float b_to_f_cached( uint8_t b ) const
	{
		return f_lookup[b];
	}

	void set_similarity( int a, int b, float sim )
	{
		//wnSimilarities.set( a, b, f_to_b(sim) );
	}

public:
	bool empty() const { return matrix.empty(); }
	size_t count( const uint8_t val ) const { /*return wnSimilarities.count(val);*/ }

	struct Row
	{
		size_t offset, from, to;
	};
	vector<Row> matrix;
	vector<uint8_t> values;

	void clear()
	{
		matrix = vector<Row>();
		values = vector<uint8_t>();
	}

	void calculate_matrix( wnb::wordnet& wn, const bool verbose=false )
	{
		std::pair<boost::adjacency_list<>::vertex_iterator,
			boost::adjacency_list<>::vertex_iterator> vs = boost::vertices(wn.wordnet_graph);

		std::vector<wnb::synset> synsets;	// should really figure out how to do this without copying all to a vector first
		for( auto it=vs.first; it!=vs.second; ++it )
			synsets.emplace_back( wn.wordnet_graph[*it] );

		//synsets.resize(DBG_SYN); // DEBUG

		wnb::nltk_similarity nltkSim( wn );

		if( verbose ) std::cout << "Generate " << endl;

		const size_t fullMatrixSize = (pow(synsets.size(),2)+synsets.size())/2;
		
		matrix.resize( synsets.size() );
		values.resize( fullMatrixSize );

		auto pos = [&synsets]( const size_t a, const size_t b )
		{
			return ((a*synsets.size())+b) - (((a*a)+a)/2);
		};

		atomic<size_t> progress(0);
		auto genfunc = [&]( const tbb::blocked_range<size_t> &range )
		{
			for( size_t _a=range.begin(); _a<range.end(); ++_a )
			{
				const wnb::synset &a = *(synsets.begin()+_a);
				const size_t n = synsets.size();

				for( size_t _b=_a; _b<synsets.size(); ++_b )
				{
					const wnb::synset &b = *(synsets.begin()+_b);

					const size_t _v = pos(_a,_b);
					auto v = values.begin() + _v;

					*v = nltk_cache::f_to_b( nltkSim(a,b) );
					//if( *v>0.32 && *v<0.34 ) *v = -1.f;
				}

				/* === find area of iterest in sim values (i.e. !=-1) ====== */
				const auto _begin = values.begin() + pos(_a,_a);
				auto begin=_begin;
				auto end = values.begin() + pos(_a,synsets.size());

				begin++; // ignore a=b comparision, always 1.0
				while( begin!=end && *begin==nullsim ) ++begin;

				--end; // neg then add 1 to avoid having to neg 1 in each loop iteraton;
				while( end>begin && *end==nullsim  ) --end;
				++end;
				/* === end ====== */

				matrix[_a] = { 0, _a+(begin-_begin), _a+(end-_begin) };

				++progress;
				if( progress % 100 == 0 ) cout << ++progress << "     \r" << flush;
			}
		};
		tbb::parallel_for( tbb::blocked_range<size_t>( 0, synsets.size() ), genfunc );
		//genfunc( tbb::blocked_range<size_t>( 0, synsets.size() ) );
		cout << endl;

		// === move all areas of interest to form a single contiguos block ======
		auto runningOffset = values.begin();
		for( size_t _a=0; _a<synsets.size(); ++_a )
		{
			auto &r = matrix[_a];
			const auto begin = values.begin()+pos(_a,r.from);
			const auto end = values.begin()+pos(_a,r.to);

			r.offset = runningOffset - values.begin();

			move( begin, end, runningOffset );
			runningOffset += end - begin;
		}
		values.erase( runningOffset, values.end() );
		cout << "pre shrink" << endl << flush;
		values.shrink_to_fit();
		cout << "post shrink" << endl << flush;
		// === end ======

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

	bool save( const std::string &path, const bool verbose=true ) const
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

			file.write( (char*)&row.offset, sizeof(row.offset) );
			file.write( (char*)&row.from, sizeof(row.from) );
			file.write( (char*)&row.to, sizeof(row.to) );
		}

		file.write( (char*)values.data(), sizeof(uint8_t)*values.size() );
		file.close();

		if( verbose ) std::cout << std::endl;
		
		return false;
	}

	bool load( const std::string &path, const bool verbose=true )
	{
		std::ifstream file( path+similaritiesFilename, std::ios::binary );
		if( !file.good() ) return true;

		clear(); // make sure the matrix is empty first
		
		// read the minimum row number and the number of rows
		size_t rowsNum;
		file.read( (char*)&rowsNum, sizeof(rowsNum) );

		matrix.resize( rowsNum );
		
		for( auto &row : matrix )
		{
			file.read( (char*)&row.offset, sizeof(row.offset) );
			file.read( (char*)&row.from, sizeof(row.from) );
			file.read( (char*)&row.to, sizeof(row.to) );
		}

		const size_t total  = matrix.back().offset + matrix.back().to - matrix.back().from;
		values.resize( total );

		file.read( (char*)values.data(), sizeof(uint8_t)*total );
		
		/*if( verbose )
		{
	#if defined(BOOST)
			boost::progress_display show_progress( total, output );
	#else
			size_t show_progress = 0;
	#endif
			for( Row &r : *this )
				show_progress += read_row( r, defaultValue );
		}
		else
			for( Row &r : *this )
				read_row( r, defaultValue );*/

		file.close();

		return false;
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

	typedef std::vector< wnb::synset > Synsets;

	inline float fast_lookup( const wnb::synset &a, const wnb::synset &b ) const
	{
		//cout << "fast_lookup(" << a.id << ", " << b.id << ")" << endl;
		if( a.id < b.id )
		{
			//cout << "  a.id < b.id" << endl;
			const auto &row = matrix[a.id];
			if( b.id < row.from || b.id >= row.to ) { /*cout << "  null" << endl;*/ return -1.f; }

			/*cout << "  row.offset = " << row.offset << endl;
			cout << "  row.from   = " << row.from << endl;
			cout << "  row.to     = " << row.to << endl;*/

			const size_t pos = row.offset + b.id - row.from;
			return b_to_f_cached( values[pos] );
		}
		else if( a.id > b.id )
		{
			//cout << "  a.id > b.id" << endl;
			const auto &row = matrix[b.id];
			if( a.id < row.from || a.id >= row.to ) return -1.f;

			const size_t pos = row.offset + a.id - row.from;
			return b_to_f_cached( values[pos] );
		}

		return 1.f;
	}

	float operator()( const wnb::synset &a, const wnb::synset &b ) const
	{
		if( empty() ) return b_to_f_cached(nullsim);

		return fast_lookup( a, b );
	}

	nltk_cache() /*: wnSimilarities( f_to_b(-1.f) )*/
	{
		for( int i=0; i<256; ++i )
			f_lookup[i] = b_to_f( (uint8_t)i );
	}

	nltk_cache( const std::string &path, const bool verbose=false ) : nltk_cache()
	{
		load( path, verbose );
	}

	nltk_cache( wnb::wordnet &wn, const bool verbose=false ) : nltk_cache()
	{
		calculate_matrix( wn, verbose );
	}
};

#endif