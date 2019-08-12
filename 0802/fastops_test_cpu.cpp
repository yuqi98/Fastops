#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "Fastops_cpu.h"

using namespace mkldnn;

using namespace std;

using namespace chrono;

using namespace Fastops;



int main(int argc, char **argv) {
    try {

        
        vector<primitive> net;
        vector<unordered_map<int, memory>> net_args;
        
        
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch())
                             .count();
        
        int times = 100;
        int n = 32;

        int batch = 1;
        Tensor_cpu mynet;
        
        std::vector<float> input_image(batch*3*n*n);
        
        srand(time(NULL));
        float min = -1;
        float max = 1;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < n; j++)
                for(int k = 0; k < n; k++)
                    input_image[i+j*3+k*3*n] = min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
        
        vector<int> input_size(4);
        input_size[0] = batch;
        input_size[1] = 3;
        input_size[2] = n;
        input_size[3] = n;
        
        
        mynet.set_data(input_image,input_size);
        vector<float> now_input = mynet.get_data();
        for(int i = 0; i < 10; i++)
            cout<<now_input[i]<<" ";
        cout<<endl;
        
        Tensor_cpu mynet2;
        mynet2.set_data(input_image,input_size);
        
        mynet.element_product(mynet2);
        vector<float> now_input2 = mynet.get_data();
        for(int i = 0; i < 10; i++)
            cout<<now_input2[i]<<" ";
        cout<<endl;
        
        mynet.element_add(mynet2);
        
        vector<float> now_input3 = mynet.get_data();
        for(int i = 0; i < 10; i++)
            cout<<now_input3[i]<<" ";
        cout<<endl;
        
        mynet.convolution_layer(net, net_args, 64, 5, 1, 2);
        mynet.relu(net, net_args);

        for(int i = 1; i < 10; i++)
        {
            mynet.convolution_layer(net, net_args, 64, 5, 1, 2);
            mynet.relu(net, net_args);
        }

        auto t1 = high_resolution_clock::now();
        for (int j = 0; j < times; ++j) {
            assert(net.size() == net_args.size() && "something is missing");
            for (size_t i = 0; i < net.size(); ++i)
                net.at(i).execute(mynet.s, net_args.at(i));
        }
        auto t2 = high_resolution_clock::now(); 
        auto duration = duration_cast<milliseconds>(t2 - t1); 
        
        cout<<duration.count()<<endl;


        mynet.s.wait();
        
        auto end = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch())
                           .count();

        vector<float> results = mynet.get_data();
        cout << "Use time " << (end - begin) / (times + 0.0) << "\n";
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
