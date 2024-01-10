#include <gtest/gtest.h>

#include <string>
#include <thread>

#include "io/onnx_serializer.h"
#include "ir/model.h"
#include "ir/node_shape_manager.h"
#include "utils/logger.h"
#include "utils/utils.h"

using namespace simple_ai;
using namespace simple_ai::utils;

TEST(IOTest, IRTest) {
    using namespace simple_ai;
    using namespace simple_ai::utils;
    using namespace simple_ai::ir;
    using namespace simple_ai::io;

    set_logger_level(LOG_LEVEL_VERBOSE);
    auto ret = init_logger("simple_ai", "./log");

    LOG_INFO("Program begins.......");

    NodeShapeManager::instance()->register_all_infer();

    std::shared_ptr<Model> model;
    Status status = OnnxSerializer::load_from_file("/home/lijian/code/simple_ai/tests/data/resnet50.onnx", model);
    if (!status.is_ok()) {
        LOG_INFO("load onnx file failed");
        std::cout << status << std::endl;
    } else {
        LOG_INFO("load onnx file successfully");
    }

    LOG_INFO("Model info:");
    LOG_INFO("domain: %s", model->get_domain().c_str());
    LOG_INFO("ir version: %d", static_cast<int>(model->get_ir_version()));
    LOG_INFO("model version: %d", static_cast<int>(model->get_model_version()));
    LOG_INFO("producer name: %s", model->get_producer_name().c_str());
    LOG_INFO("producer version: %s", model->get_producer_version().c_str());

    auto graph = model->get_graph();

    LOG_INFO("\n\nonnx order nodes:\n");
    const auto& nodes = graph->get_nodes();
    int i = -1;
    for(const auto& node : nodes){
        ++i;
        LOG_INFO("node %d name: %s", i, node->name().c_str());
    }

    LOG_INFO("start topological sort");
    status = graph->construct_topology();
    LOG_INFO("topological sorting returns: %s", status.to_string().c_str());

    LOG_INFO("topological nodes:\n");
    const auto& topological_nodes = graph->get_topological_nodes();
    i = -1;
    for(const auto& topo_node : topological_nodes){
        ++i;
        LOG_INFO("node %d name: %s", i, topo_node->name().c_str());
        LOG_INFO("\toutput shape: %s", topo_node->output_args()[0]->shape().to_string().c_str());
    }

    std::unordered_map<std::string, int> node_type_stats;
    for(auto& node : nodes){
        auto ret = node_type_stats.emplace(node->type(), 1);
        if(!ret.second){
            ret.first->second++;
        }
    }

    LOG_INFO("Node Statistics: ");
    for(auto& pair : node_type_stats){
        LOG_INFO("%s : %d", pair.first.c_str(), pair.second);
    }

    LOG_INFO("Program ends.......");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    release_logger();
}