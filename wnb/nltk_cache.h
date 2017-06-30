#if !defined(NLTK_CACHE_H)
#define NLTK_CACHE_H

#include <atomic>
#include <vector>
#include <iterator>
#include <fstream>

#include "wnb/core/wordnet.hh"
#include "wnb/nltk_similarity.hh"

#include "tbb/parallel_for_each.h"

#include <boost/progress.hpp>

class nltk_cache
{
public:
	const std::string similaritiesFilename = "similarities";
	static const uint8_t nullsim = 255;

	static uint8_t f_to_b( float f )
	{
		return round( pow( f, -1.f ) );
	}

	static float b_to_f( uint8_t b )
	{
		return b == nullsim ? -1.f : pow( b, -1.f );
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
	//size_t count( const uint8_t val ) const { /*return wnSimilarities.count(val);*/ }

	struct Row
	{
		size_t offset, from, to;
	};
	std::vector<Row> matrix;
	std::vector<uint8_t> values;

	void clear()
	{
		matrix = std::vector<Row>();
		values = std::vector<uint8_t>();
	}

	void calculate_matrix( wnb::wordnet& wn, const bool verbose=false )
	{
		auto synsets = wn.get_synsets();
		wnb::nltk_similarity nltkSim( wn );

		std::unique_ptr<boost::progress_display> showProgress;
		std::mutex lockProgress;
		if( verbose )
		{
			std::cout << std::endl << "### Generating similarity cache";
			showProgress = std::make_unique<boost::progress_display>( synsets.size() );
		}

		const size_t fullMatrixSize = (pow(synsets.size(),2)+synsets.size())/2;
		matrix.resize( synsets.size() );
		values.resize( fullMatrixSize );

		/* calculate the storage position for synset pair given synset ids */
		auto pos = [&synsets]( const size_t a, const size_t b )
		{
			return ((a*synsets.size())+b) - (((a*a)+a)/2);
		};

		/* generate all similarity values in given range */
		auto genfunc = [&]( const tbb::blocked_range<size_t> &range )
		{
			for( size_t _a=range.begin(); _a!=range.end(); ++_a )
			{
				const wnb::synset &a = *(synsets.begin()+_a);
				const size_t n = synsets.size();

				for( size_t _b=_a; _b<synsets.size(); ++_b )
				{
					const wnb::synset &b = *(synsets.begin()+_b);

					const size_t _v = pos(_a,_b);
					auto v = values.begin() + _v;

					*v = nltk_cache::f_to_b( nltkSim(a,b) );
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
			}

			if( verbose )
			{
				std::lock_guard<std::mutex> lock( lockProgress );
				*showProgress += range.end() - range.begin();
			}
		};
		tbb::parallel_for( tbb::blocked_range<size_t>( 0, synsets.size() ), genfunc );
		//genfunc( tbb::blocked_range<size_t>( 0, synsets.size() ) ); // single thread test

		if( verbose ) std::cout << std::endl << "calculated: " << values.size() << std::endl;

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
		values.shrink_to_fit();
		// === end ======

		if( verbose ) std::cout << "kept: " << values.size() << std::endl;
	}

	/* writes the currect cache as a binary file to filename.
		returns true if error, false if success.

		format is:
			[number of rows = n] size_t         

			[offset into cache where row 0 data can be found] size_t
			[minimum column in row 0 that is actually stored] size_t
			[maximum column in row 0 that is actually stored] size_t

			[offset into cache where row 1 data can be found] size_t
			[minimum column in row 1 that is actually stored] size_t
			[maximum column in row 1 that is actually stored] size_t

			...

			[offset into cache where row n-1 data can be found] size_t
			[minimum column in row n-1 that is actually stored] size_t
			[maximum column in row n-1 that is actually stored] size_t

			[contiguous block of v similarity values] uint8_t * v
	*/

	bool save( const std::string &path, const bool verbose=true ) const
	{
		if( verbose) std::cout << "Save file" << std::endl;

		std::ofstream file( path+similaritiesFilename, std::ios::binary );
		if( !file.good() ) return true;

		// write the minimum row number and the number of rows
		const size_t total = values.size();
		const size_t rowsNum = matrix.size();
		file.write( (char*)&rowsNum, sizeof(rowsNum) );

		size_t progress = 0;

		for( const auto &row : matrix )
		{
			if( verbose )
				std::cout << ++progress << "\r" << std::flush;
			
			file.write( (char*)&row.offset, sizeof(row.offset) );
			file.write( (char*)&row.from, sizeof(row.from) );
			file.write( (char*)&row.to, sizeof(row.to) );
		}

		file.write( (char*)values.data(), sizeof(decltype(values)::value_type)*values.size() );
		file.close();

		if( verbose ) std::cout << std::endl;
		
		return false;
	}

	bool load( const std::string &path, const bool verbose=true )
	{
		std::ifstream file( path+similaritiesFilename, std::ios::binary );
		if( !file.good() ) return true;

		std::unique_ptr<boost::progress_display> showProgress;
		if( verbose )
		{
			std::cout << std::endl << "### Loading similarity cache";
			showProgress = std::make_unique<boost::progress_display>( 100 );
		}

		clear(); // make sure the matrix is empty first
		
		if( verbose ) *showProgress += 1;

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

		if( verbose ) *showProgress += 1;

		const size_t total  = matrix.back().offset + matrix.back().to - matrix.back().from;
		values.resize( total );

		if( verbose ) *showProgress += 1;

		file.read( (char*)values.data(), sizeof(decltype(values)::value_type)*total );

		if( verbose )
		{
			*showProgress += 97;
			std::cout << "cache_vals: " << values.size() << std::endl;
			showProgress.release();
		}

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

	inline float fast_lookup( const wnb::synset &a, const wnb::synset &b ) const
	{
		// better performance unrolled than when using min/max
		if( a.id < b.id )
		{
			const auto &row = matrix[a.id];
			if( b.id < row.from || b.id >= row.to ) return -1.f;

			const size_t pos = row.offset + b.id - row.from;
			return b_to_f_cached( values[pos] );
		}
		else if( a.id > b.id )
		{
			const auto &row = matrix[b.id];
			if( a.id < row.from || a.id >= row.to ) return -1.f;

			const size_t pos = row.offset + a.id - row.from;
			return b_to_f_cached( values[pos] );
		}

		return 1.f;
	}

	float operator()( const wnb::synset &a, const wnb::synset &b ) const
	{
		if( empty() || std::max(a.id,b.id) >= matrix.size() ) return b_to_f_cached(nullsim);

		return fast_lookup( a, b );
	}

	nltk_cache()
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