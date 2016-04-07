#include <wnb/core/wordnet.hh>
#include <wnb/std_ext.hh>

#include <string>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/filtered_graph.hpp>

namespace wnb
{

  //FIXME: Make (smart) use of fs::path
  wordnet::wordnet(const std::string& wordnet_dir, bool verbose)
    : _verbose(verbose)
  {
    if (_verbose)
    {
      std::cout << wordnet_dir << std::endl;
    }

    info = preprocess_wordnet(wordnet_dir);

    wordnet_graph = graph(info.nb_synsets());
    load_wordnet(wordnet_dir, *this, info);

    if (_verbose)
    {
      std::cout << "nb_synsets: " << info.nb_synsets() << std::endl;
    }
    //FIXME: this check is only valid for Wordnet 3.0
    assert(info.nb_synsets() == 142335);//117659);
  }

  std::vector<synset>
  wordnet::get_synsets(const std::string& word, pos_t pos)
  {
    std::vector<synset> synsets;

    // morphing
    std::string mword = morphword(word, pos);
    if (mword == "")
      return synsets;

    // binary_search
    typedef std::vector<index> vi;
    std::pair<vi::iterator,vi::iterator> bounds = get_indexes(mword);

    vi::iterator it;
    for (it = bounds.first; it != bounds.second; it++)
    {
      if (pos == pos_t::UNKNOWN || it->pos == pos)
      {
        for (std::size_t i = 0; i < it->synset_ids.size(); i++)
        {
          int id = it->synset_ids[i];
          synsets.push_back(wordnet_graph[id]);
        }
      }
    }

    return synsets;
  }

  std::pair<std::vector<index>::iterator, std::vector<index>::iterator>
  wordnet::get_indexes(const std::string& word)
  {
    index light_index;
    light_index.lemma = word;

    typedef std::vector<index> vi;
    std::pair<vi::iterator,vi::iterator> bounds =
      std::equal_range(index_list.begin(), index_list.end(), light_index);

    return bounds;
  }

  std::string
  wordnet::wordbase(const std::string& word, int ender)
  {
    if (ext::ends_with(word, info.sufx[ender]))
    {
      int sufxlen = info.sufx[ender].size();
      std::string strOut = word.substr(0, word.size() - sufxlen);
      if (!info.addr[ender].empty())
        strOut += info.addr[ender];
      return strOut;
    }
    return word;
  }

  bool is_defined(const std::string& word, pos_t pos)
  {
    // hack FIXME: Some verbs are built with -e suffix ('builde' is just an example).
    if (pos == V && word == "builde")
      return false;
    return true;
  }

  // Try to find baseform (lemma) of individual word in POS
  std::string
  wordnet::morphword(const std::string& word, pos_t pos)
  {
    std::vector<std::string> results;

    auto it = morphologicalrules.find( pos );
    if( it != morphologicalrules.end() )
    {
      results = _morphword( word, pos );
    }
    else
    {
      for( auto &pos : morphologicalrules )
        if( pos.first != pos_t::S )     // revisit, probably filter if
        {
          std::vector<std::string> temp = _morphword( word, pos.first ); // gotta be a better way than this
          results.insert( results.end(), temp.begin(), temp.end() );
        }
    }

    if( results.empty() ) return "";

    //cout << "results: ";
    //for( auto r : results ) cout << r << ", "; cout << endl;

    return *(results.begin());
  }

  std::vector<std::string> 
  wordnet::_morphword(const std::string &form, pos_t pos)
  {
    std::vector<std::pair<std::string,std::string> > &morphsubs = morphologicalrules[pos];
    std::vector<std::string> results;

    auto copyfilter = [this,pos]( const std::string &s ) 
          { 
            auto indexes = this->get_indexes( s );
            for( auto it=indexes.first; it!=indexes.second; ++it )
              if( it->pos == pos )
                return true;

            return false;
          };

    // check the exceptions list
    std::map<std::string,std::vector<std::string> > &exceptions = exc[pos];
    std::map<std::string,std::vector<std::string> >::iterator except = exceptions.find( form  );
    if( except != exceptions.end() )
    {
      results.resize( except->second.size() );
      results.erase( copy_if( except->second.begin(), except->second.end(), results.begin(), copyfilter ), results.end() );

      return results;
    }

    // apply the modify rules
    std::vector<std::string> forms { form }, rules;
    do
    {
      rules.clear();
      for( std::string &f : forms )
      {
        for( std::pair<std::string,std::string> &sub : morphsubs )
        {
          if( f.length() > sub.first.length() &&
            !f.compare( f.length() - sub.first.length(), sub.first.length(), sub.first ) )
          {
            rules.emplace_back( f.substr( 0, f.length() - sub.first.length() ) + sub.second );
          }
        }
      }

      // filter
      results.clear();
      copy_if( forms.begin(), forms.end(), back_inserter(results), copyfilter );
      copy_if( rules.begin(), rules.end(), back_inserter(results), copyfilter );

      if( !results.empty() )
      {
        return results;
      }

      //forms = rules;
      forms.swap( rules );
    }
    while( !forms.empty() );

    // can't find anything
    return {};
  }

  

} // end of namespace wnb
