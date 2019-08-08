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
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch())
                             .count();
        int times = 100;
        int n = 32;

        int batch = 1;
        Data_cpu mynet;

        mynet.init(argc, argv);

        std::vector<float> input_image(batch*3*n*n);

        mynet.set_data(input_image,batch,3,n,n);

        mynet.convolution_layer(batch, 3, n, n, 64, 5, 1, 2);
        mynet.relu(batch,64,n,n,64);

        for(int i = 1; i < 10; i++)
        {
            mynet.convolution_layer(batch, 64, n, n, 64, 5, 1, 2);
            mynet.relu(batch,64,n,n,64);
        }

        auto t1 = high_resolution_clock::now();
        for (int j = 0; j < times; ++j) {
            assert(mynet.net.size() == mynet.net_args.size() && "something is missing");
            for (size_t i = 0; i < mynet.net.size(); ++i)
                mynet.net.at(i).execute(mynet.s, mynet.net_args.at(i));
        }
        auto t2 = high_resolution_clock::now(); 
        auto duration = duration_cast<milliseconds>(t2 - t1); 
        
        cout<<duration.count()<<endl;


        mynet.s.wait();
        
        auto end = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch())
                           .count();

        vector<float> results = mynet.get_data(mynet.current);
        cout << "Use time " << (end - begin) / (times + 0.0) << "\n";
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
