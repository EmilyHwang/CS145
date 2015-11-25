// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the multiclass classification tools  
    from the dlib C++ Library.  Specifically, this example will make points from 
    three classes and show you how to train a multiclass classifier to recognize 
    these three classes.

    The classes are as follows:
        - class 1: points very close to the origin
        - class 2: points on the circle of radius 10 around the origin
        - class 3: points that are on a circle of radius 4 but not around the origin at all
*/

#include <dlib/svm_threaded.h>

#include <cstdio>
#include <iostream>

#include <vector>

#include "json/json.h"
#include <string>
#include <fstream>

#include <map>

#include <dlib/rand.h>

#include "stemming/english_stem.h"

#include <ctime>

using namespace std;
using namespace dlib;

// Our data will be 2-dimensional data. So declare an appropriate type to contain these points.
typedef matrix<double> sample_type;

// ----------------------------------------------------------------------------------------

void parse_data(std::vector<sample_type>& samples, std::vector<string>& labels);
void get_test_data(std::vector<sample_type>& samples, std::vector<string>& ids);

struct val_greater_than : binary_function < pair<string,int>, pair<string,int>, bool > {
	bool operator() (const pair<string,int>& x, const pair<string,int>& y) const
	{
		return x.second>y.second;
	}
}val_gt;

// ----------------------------------------------------------------------------------------

int main()
{
    try
    {
		//samples to return
		std::vector<sample_type> samples;
		std::vector<string> labels;
		parse_data(samples, labels);

        cout << "samples.size(): "<< samples.size() << endl;

        // The main object in this example program is the one_vs_one_trainer.  It is essentially 
        // a container class for regular binary classifier trainer objects.  In particular, it 
        // uses the any_trainer object to store any kind of trainer object that implements a 
        // .train(samples,labels) function which returns some kind of learned decision function.  
        // It uses these binary classifiers to construct a voting multiclass classifier.  If 
        // there are N classes then it trains N*(N-1)/2 binary classifiers, one for each pair of 
        // labels, which then vote on the label of a sample.
        //
        // In this example program we will work with a one_vs_one_trainer object which stores any 
        // kind of trainer that uses our sample_type samples.
        typedef one_vs_all_trainer<any_trainer<sample_type>, string> ovo_trainer;


        // Finally, make the one_vs_one_trainer.
        ovo_trainer trainer;

        // Next, we will make two different binary classification trainer objects.  One
        // which uses kernel ridge regression and RBF kernels and another which uses a
        // support vector machine and polynomial kernels.  The particular details don't matter.
        // The point of this part of the example is that you can use any kind of trainer object
        // with the one_vs_one_trainer.
        typedef linear_kernel<sample_type> linear_kernel;

        // make the binary trainers and set some parameters
        svm_c_linear_trainer<linear_kernel> linear_trainer;
        ///poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
        linear_trainer.set_c(5);

		trainer.set_trainer(linear_trainer);
        // Now tell the one_vs_one_trainer that, by default, it should use the rbf_trainer
        // to solve the individual binary classification subproblems.
       // trainer.set_trainer(rbf_trainer);
        // We can also get more specific.  Here we tell the one_vs_one_trainer to use the
        // poly_trainer to solve the class 1 vs class 2 subproblem.  All the others will
        // still be solved with the rbf_trainer.
        //trainer.set_trainer(poly_trainer, "greek, 2);

        // Now let's do 5-fold cross-validation using the one_vs_one_trainer we just setup.
//cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
        // The output is shown below.  It is the confusion matrix which describes the results.  Each row 
        // corresponds to a class of data and each column to a prediction.  Reading from top to bottom, 
        // the rows correspond to the class labels if the labels have been listed in sorted order.  So the
        // top row corresponds to class 1, the middle row to class 2, and the bottom row to class 3.  The
        // columns are organized similarly, with the left most column showing how many samples were predicted
        // as members of class 1.
        // 
        // So in the results below we can see that, for the class 1 samples, 60 of them were correctly predicted
        // to be members of class 1 and 0 were incorrectly classified.  Similarly, the other two classes of data
        // are perfectly classified.
        
        //    cross validation: 
        //    60  0  0 
        //    0 70  0 
        //    0  0 80 
    

        // Next, if you wanted to obtain the decision rule learned by a one_vs_one_trainer you 
        // would store it into a one_vs_one_decision_function.
        one_vs_all_decision_function<ovo_trainer> df = trainer.train(samples, labels);

        cout << "predicted label: "<< df(samples[0])  << ", true label: "<< labels[0] << endl;
        cout << "predicted label: "<< df(samples[90]) << ", true label: "<< labels[90] << endl;
        // The output is:
        /*
            predicted label: 2, true label: 2
            predicted label: 1, true label: 1
        */


        // If you want to save a one_vs_one_decision_function to disk, you can do
        // so.  However, you must declare what kind of decision functions it contains. 
        one_vs_all_decision_function<ovo_trainer, 
        decision_function<linear_kernel>  // This is the output of the poly_traine
        > df2, df3;


        // Put df into df2 and then save df2 to disk.  Note that we could have also said
        // df2 = trainer.train(samples, labels);  But doing it this way avoids retraining.
        df2 = df;
        serialize("df.dat") << df2;

        // load the function back in from disk and store it in df3.  
        deserialize("df.dat") >> df3;


        // Test df3 to see that this worked.
        cout << endl;
        cout << "predicted label: "<< df3(samples[0])  << ", true label: "<< labels[0] << endl;
        cout << "predicted label: "<< df3(samples[90]) << ", true label: "<< labels[90] << endl;
        // Test df3 on the samples and labels and print the confusion matrix.
        cout << "test deserialized function: \n" << test_multiclass_decision_function(df3, samples, labels) << endl;


		//need to read test data
		//samples to return
		std::vector<sample_type> test_samples;
		std::vector<string> test_ids;
		get_test_data(test_samples, test_ids);

		ofstream results;
		results.open("submission.csv");
		results << "id,cuisine\n";
		for(size_t i = 0; i < test_ids.size(); ++i)
		{
			results << test_ids[i] << "," << df3(test_samples[i]) << "\n";
		}

		results.close();

		time_t rawtime;
  struct tm * timeinfo;
  char buffer[80];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,80,"%d-%m-%Y %I:%M:%S",timeinfo);
  std::string str(buffer);

  std::cout << "time completed: " << str << endl;

        // Finally, if you want to get the binary classifiers from inside a multiclass decision
        // function you can do it by calling get_binary_decision_functions() like so:
       /* one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
        functs = df.get_binary_decision_functions();
        cout << "number of binary decision functions in df: " << functs.size() << endl;
        // The functs object is a std::map which maps pairs of labels to binary decision
        // functions.  So we can access the individual decision functions like so:
        //decision_function<poly_kernel> df_1_2 = any_cast<decision_function<poly_kernel> >(functs[make_unordered_pair(1,2)]);
        decision_function<rbf_kernel>  df_1_3 = any_cast<decision_function<rbf_kernel>  >(functs[make_unordered_pair(1,3)]);
        // df_1_2 contains the binary decision function that votes for class 1 vs. 2.
        // Similarly, df_1_3 contains the classifier that votes for 1 vs. 3.

        // Note that the multiclass decision function doesn't know what kind of binary
        // decision functions it contains.  So we have to use any_cast to explicitly cast
        // them back into the concrete type.  If you make a mistake and try to any_cast a
        // binary decision function into the wrong type of function any_cast will throw a
        // bad_any_cast exception.*/
    }
    catch (std::exception& e)
    {
        cout << "exception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------
void parse_data(std::vector<sample_type>& samples, std::vector<string>& labels)
{

	Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    ifstream file("train.json", ifstream::binary);

    bool parsedSuccess = reader.parse(file, root, false);

    if (!parsedSuccess) {
        cout << "Failed to parse JSON\n"
			<< reader.getFormattedErrorMessages();
		return;
    }

	std::map<std::string, size_t> frequency_map;
    for (size_t i = 0; i < root.size(); i++) {

        string cuisine = root[i].get("cuisine", "null").asString();
		const Json::Value ingredients = root[i].get("ingredients", "null");
		for (size_t j = 0; j < ingredients.size(); ++j)
		{
			string ingredient_string= ingredients[j].asString();

			//lower case everything
			//std::transform(ingredient_string.begin(), ingredient_string.end(), ingredient_string.begin(), ::tolower);

			//remove punctuation
			//ingredient_string.erase(std::remove_if(ingredient_string.begin(), ingredient_string.end(), ispunct), ingredient_string.end());

			//only take last two words
			/*size_t found1 = ingredient_string.find_last_of( " " );
			stringstream sstr(ingredient_string);
			std::vector<string> vstr;

			while(sstr >> ingredient_string)
			{
			  vstr.push_back(ingredient_string);
			}

			if(vstr.size() >= 2)
				ingredient_string = vstr[vstr.size()-2] + " " + vstr[vstr.size()-1];
			else
				ingredient_string = vstr[0];

			//Stemming
			wchar_t* UnicodeTextBuffer = new wchar_t[ingredient_string.length()+1];
			std::wmemset(UnicodeTextBuffer, 0, ingredient_string.length()+1);
			std::mbstowcs(UnicodeTextBuffer, ingredient_string.c_str(), ingredient_string.length());
			std::wstring wingredient = UnicodeTextBuffer;

			stemming::english_stem<> StemEnglish;
			StemEnglish(wingredient);

			ingredient_string = string(wingredient.begin(), wingredient.end());*/

			if (frequency_map.find(ingredient_string) == frequency_map.end())
			{
				frequency_map[ingredient_string] = 1;
			} 
			else 
			{
				int x = frequency_map[ingredient_string];
				frequency_map[ingredient_string] = x + 1;
			}
		}

		labels.push_back(cuisine);
    }
	cout << "number of recipes: " << labels.size() << endl;
	cout << "number of ingredients: " << frequency_map.size() << endl;

	std::vector<pair<string, int>> mapcopy(frequency_map.begin(), frequency_map.end());
	sort(mapcopy.begin(), mapcopy.end(), val_gt);

	mapcopy.resize(std::min<size_t>(mapcopy.size(), 2000));
	
	std::map<std::string, size_t> ingredients_map;
	for(size_t i = 0; i < mapcopy.size(); ++i)
	{
		ingredients_map[mapcopy[i].first] = ingredients_map.size();
	}

	cout << "number of ingredients: " << ingredients_map.size() << endl;
	cout << "root size: " << root.size();
	cout << "label size: " << labels.size();
	samples.reserve(labels.size());

	for(size_t i = 0; i < root.size(); ++i) {
		const Json::Value ingredients = root[i].get("ingredients", "null");

		matrix<double> sample;
		sample.set_size(ingredients_map.size(), 1);

		sample = 0;

		for (size_t j = 0; j < ingredients.size(); ++j)
		{
			string ingredient_string = ingredients[j].asString();
			if (ingredients_map.find(ingredient_string) != ingredients_map.end())
			{
				int index = ingredients_map[ingredient_string];
				cout << "index to put: " << index << "for i= " << i << endl;
				sample(index) = 1;
			}
		}

		samples.push_back(sample);
	}

	file.close();
	cout << "number of samples: " << samples.size() << endl;
}

// ----------------------------------------------------------------------------------------

void get_test_data(std::vector<sample_type>& samples, std::vector<string>& ids)
{

	Json::Value root;   // will contains the root value after parsing.
    Json::Reader reader;
    ifstream file("test.json", ifstream::binary);

    bool parsedSuccess = reader.parse(file, root, false);

    if (!parsedSuccess) {
        cout << "Failed to parse JSON\n"
			<< reader.getFormattedErrorMessages();
		return;
    }

	std::map<std::string, size_t> frequency_map;
    for (size_t i = 0; i < root.size(); i++) {

        string id = root[i].get("id", "null").asString();
		const Json::Value ingredients = root[i].get("ingredients", "null");
		for (size_t j = 0; j < ingredients.size(); ++j)
		{
			string ingredient_string= ingredients[j].asString();

			if (frequency_map.find(ingredient_string) == frequency_map.end())
			{
				frequency_map[ingredient_string] = 1;
			} 
			else 
			{
				int x = frequency_map[ingredient_string];
				frequency_map[ingredient_string] = x + 1;
			}
		}

		ids.push_back(id);
    }
	cout << "number of recipes: " << ids.size() << endl;
	cout << "number of ingredients: " << frequency_map.size() << endl;

	std::vector<pair<string, int>> mapcopy(frequency_map.begin(), frequency_map.end());
	sort(mapcopy.begin(), mapcopy.end(), val_gt);

	mapcopy.resize(std::min<size_t>(mapcopy.size(), 2000));
	
	std::map<std::string, size_t> ingredients_map;
	for(size_t i = 0; i < mapcopy.size(); ++i)
	{
		ingredients_map[mapcopy[i].first] = ingredients_map.size();
	}

	cout << "number of ingredients: " << ingredients_map.size() << endl;
	samples.reserve(ids.size());

	for(size_t i = 0; i < root.size(); ++i) {
		const Json::Value ingredients = root[i].get("ingredients", "null");

		matrix<double> sample;
		sample.set_size(ingredients_map.size(), 1);

		sample = 0;

		for (size_t j = 0; j < ingredients.size(); ++j)
		{
			string ingredient_string = ingredients[j].asString();
			if (ingredients_map.find(ingredient_string) != ingredients_map.end())
			{
				int index = ingredients_map[ingredient_string];
				//cout << "index to put: " << index << "for i= " << i << endl;
				sample(index) = 1;
			}
		}

		samples.push_back(sample);
	}

	file.close();
	cout << "finished test data- number of samples: " << samples.size() << endl;
}

// ----------------------------------------------------------------------------------------

