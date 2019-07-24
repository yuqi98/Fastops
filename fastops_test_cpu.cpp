#include <assert.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "Fastops.h"

using namespace mkldnn;

using namespace std;

using namespace chrono;


void simple_net(int times = 100) {

    int n = 32;

    using tag = memory::format_tag;
    using dt = memory::data_type;
    engine eng(engine::kind::cpu, 0);
    stream s(eng);
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    const memory::dim batch = 1;
    //cout<<1<<endl;

    memory input_layer = Fastops::cpu_input_layer(eng, batch, 3, 64, n, n);
    //cout<<2<<endl;

    memory conv_layer = Fastops::cpu_convolution_layer_relu(net, net_args, eng, batch, 3, n, n, 64, 5, 1, 2, input_layer);
    //cout<<3<<endl;
    for(int i = 1; i < 10; i++)
    	conv_layer = Fastops::cpu_convolution_layer_relu(net, net_args, eng, batch, 64, n, n, 64, 5, 1, 2, conv_layer);


    auto t1 = high_resolution_clock::now();
    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
    }
    auto t2 = high_resolution_clock::now(); 
    auto duration = duration_cast<milliseconds>(t2 - t1); 
    
    cout<<duration.count()<<endl;


    s.wait();
}

int main(int argc, char **argv) {
    try {
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch())
                             .count();
        int times = 100;
        simple_net(times);
        auto end = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now().time_since_epoch())
                           .count();
        cout << "Use time " << (end - begin) / (times + 0.0) << "\n";
    } catch (error &e) {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
