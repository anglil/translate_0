#include "charlm.h"

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);

	string task = argv[1];
	
	charlm_lstm* charlm = new charlm_lstm();

	if (task == "train") {
    char* train_file = argv[2];
  	char* dev_file = argv[3];
  
  	unsigned num_layer = atoi(argv[4]);
  	unsigned input_dim = atoi(argv[5]);
  	unsigned hidden_dim = atoi(argv[6]);
  
    string model_file = argv[7];
  	string dict_file = argv[8];
    string ppl_file = argv[9];
    string log_file = argv[10];

		float lr = atof(argv[11]);

		charlm->train(train_file, dev_file, num_layer, input_dim, hidden_dim,	model_file,	dict_file, ppl_file, log_file, lr);	

	} else if (task == "test") {
		char* test_file = argv[2];

  	unsigned num_layer = atoi(argv[3]);
  	unsigned input_dim = atoi(argv[4]);
  	unsigned hidden_dim = atoi(argv[5]);
  
    string model_file = argv[6];
  	string dict_file = argv[7];
    string log_file = argv[8];

		charlm->test(test_file,	num_layer, input_dim,	hidden_dim,	model_file,	dict_file, log_file);
	} 
	return 0;
}
