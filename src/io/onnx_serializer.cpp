#include "io/onnx_serializer.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stack>
#include <unordered_set>

#include "utils/logger.h"

using namespace simple_ai::common;
using namespace simple_ai::utils;
using namespace simple_ai::ir;
using namespace simple_ai::framework;

namespace simple_ai {
namespace io {

Status OnnxSerializer::load_from_file(const std::string& file_path, std::shared_ptr<Model>& model_ptr) {
    auto loader = [file_path](onnx::ModelProto& onnx_model) {
        if (!file_exist(file_path)) {
            return Status(StatusCode::FILE_NOT_FOUND, "file not found: " + file_path);
        }

        std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
        if (!ifs.is_open()) {
            return Status(StatusCode::FILE_NOT_FOUND, "Open file failed: " + file_path);
        }

        google::protobuf::io::IstreamInputStream input_stream(&ifs);
        google::protobuf::io::CodedInputStream coded_input(&input_stream);
        bool parsed = onnx_model.ParseFromCodedStream(&coded_input);
        ifs.close();
        if (!parsed) {
            return Status(StatusCode::INVALID_MODEL, "Parse onnx model failed: " + file_path);
        }

        return Status::ok();
    };

    return load_with_loader(loader, model_ptr);
}

Status OnnxSerializer::load_from_memory(const void* data, size_t data_len, std::shared_ptr<Model>& model_ptr) {
    auto loader = [data, data_len](onnx::ModelProto& onnx_model) {
        if (data == nullptr || data_len == 0) {
            return Status(StatusCode::INVALID_PARAM, "Parse onnx model from memory failed, invalid parameters");
        }
        bool parsed = onnx_model.ParseFromArray(data, static_cast<int>(data_len));
        if (!parsed) {
            return Status(StatusCode::INVALID_MODEL, "Parse onnx model from memory failed");
        }

        return Status::ok();
    };

    return load_with_loader(loader, model_ptr);
}

Status OnnxSerializer::validate_onnx_proto(const onnx::ModelProto& model) {
    bool has_graph = model.has_graph();
    if (!has_graph) {
        return Status(StatusCode::INVALID_MODEL, "no graph in onnx model");
    }

    if (model.opset_import_size() == 0) {
        return Status(StatusCode::INVALID_MODEL, "opset_import missed in onnx model");
    }

    if (!onnx::Version_IsValid(model.ir_version())) {
        std::ostringstream oss;
        oss << "unsupported model IR version: " << model.ir_version();
        return common::Status(StatusCode::INVALID_MODEL, oss.str());
    }

    if (model.ir_version() < 4) {
        std::ostringstream oss;
        oss << "Too old ir version: " << model.ir_version() << ", not supported now";
        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    return Status::ok();
}

Status OnnxSerializer::load_with_loader(std::function<Status(onnx::ModelProto&)> loader,
                                        std::shared_ptr<Model>& model_ptr) {
    onnx::ModelProto onnx_model;

    // step 1. load the onnx model
    Status status = loader(onnx_model);
    if (!status.is_ok()) {
        return status;
    }

    // step 2. validate the onnx model
    status = validate_onnx_proto(onnx_model);
    if (!status.is_ok()) {
        return status;
    }

    // step 3. parse the onnx model
    model_ptr = std::make_shared<Model>();
    status = parse_onnx_model(onnx_model, model_ptr);
    if (!status.is_ok()) {
        return status;
    }

    return Status::ok();
}

Status OnnxSerializer::parse_onnx_model(const onnx::ModelProto& onnx_model, std::shared_ptr<Model>& ir_model) {
    // set metadata props
    {
        std::unordered_map<std::string, std::string> meta_map;
        for (auto& prop : onnx_model.metadata_props()) {
            meta_map[prop.key()] = prop.value();
        }

        ir_model->set_metadata(meta_map);
    }

    // set opset
    {
        std::unordered_map<std::string, int64_t> domain_version;
        for (auto& opset : onnx_model.opset_import()) {
            const auto& domain = opset.domain();
            const auto version = opset.version();
            domain_version[domain] = version;
        }

        // TODO, valida the domain and version
        ir_model->set_domain_version(domain_version);
    }

    // set functions
    {
        for (auto& func : onnx_model.functions()) {
            const auto& domain = func.domain();
            const auto& name = func.name();
        }

        // TODO, now skip the functions
    }

    ir_model->set_ir_version(onnx_model.ir_version());
    ir_model->set_producer_name(onnx_model.producer_name());
    ir_model->set_producer_version(onnx_model.producer_version());
    ir_model->set_domain(onnx_model.domain());
    ir_model->set_model_version(onnx_model.model_version());
    ir_model->set_doc_string(onnx_model.doc_string());

    auto ir_graph = std::make_unique<Graph>(*(ir_model.get()));
    Status status = parse_onnx_graph(onnx_model.graph(), ir_graph);
    if (!status.is_ok()) {
        return status;
    }

    ir_model->set_graph(std::move(ir_graph));

    return Status::ok();
}

Status OnnxSerializer::parse_onnx_graph(const onnx::GraphProto& onnx_graph, std::unique_ptr<Graph>& ir_graph) {
    std::unordered_map<std::string, NodeArg> name_to_nodearg_map;

    // Step 1. Process "Constant" nodes. Retrieve "TensorProto" attributes in the "Constant" node as a Tensor.
    for (auto& proto_node : onnx_graph.node()) {
        if (proto_node.op_type() != "Constant") {
            continue;
        }

        LOG_INFO("Constant node: %s", proto_node.name().c_str());
        std::unique_ptr<Tensor> ir_tensor;
        auto status = constant_protonode_to_tensor(proto_node, ir_tensor);
        if (status.is_ok()) {
            if (ir_graph->has_initializer(ir_tensor->name())) {
                LOG_WARNING("Tensor [%s] has already exist in the graph", ir_tensor->name().c_str());
            }
            // add. if the initializer has already existed in the graph, replace it.
            ir_graph->add_initializer(std::move(ir_tensor));
        }
    }

    // Step 2. Process the inputs name, type, shape information of the graph.
    for (auto& input : onnx_graph.input()) {
        if (!input.name().empty()) {
            LOG_INFO("Grapn input name: %s", input.name().c_str());
            const onnx::TypeProto& type = input.type();
            if (type.value_case() == onnx::TypeProto::ValueCase::kTensorType) {
                const onnx::TypeProto_Tensor tensor_type = type.tensor_type();
                PrimitiveDataType dt =
                    tensor_datatype_to_primitive(static_cast<onnx::TensorProto_DataType>(tensor_type.elem_type()));
                if (dt == PrimitiveDataType::UNKNOWN) {
                    return Status(StatusCode::INVALID_MODEL, "unsupported data type of graph inputs");
                }

                TensorShape shape = shapeproto_to_tensorshape(tensor_type.shape());
                NodeArg arg(input.name(), dt, shape);
                name_to_nodearg_map.emplace(input.name(), arg);
                ir_graph->get_or_create_nodearg(input.name(), arg);
                ir_graph->add_input_name(input.name());
            } else {
                LOG_WARNING("Graph input [%s] has no type or has an unsupported type", input.name().c_str());
            }
        } else {
            LOG_WARNING("Graph input name is empty");
        }
    }

    // Step 3. copy tensor proto to tensor ir map
    IAllocator* allocator = AllocatorManager::instance()->get_allocator(IAllocator::Type::CPU);
    for (auto& initializer : onnx_graph.initializer()) {
        std::unique_ptr<Tensor> tensor;
        auto ret = retrieve_tensor_data(initializer, tensor, allocator, initializer.name());
        if (!ret.is_ok()) {
            LOG_WARNING("Parsing initializer[%s] fails", initializer.name().c_str());
            return ret;
        }

        LOG_INFO("Initializer name: %s", tensor->name().c_str());
        NodeArg* tensor_arg = ir_graph->get_nodearg(tensor->name());
        if (tensor_arg == nullptr) {
            NodeArg arg(tensor->name(), tensor->data_type(), tensor->shape());
            name_to_nodearg_map.emplace(tensor->name(), arg);
            ir_graph->get_or_create_nodearg(tensor->name(), arg);
        } else {
            LOG_WARNING("Initializer [%s] appears in graph inputs and will not be treated as constant value",
                        tensor->name().c_str());
        }

        if (ir_graph->has_initializer(tensor->name())) {
            LOG_WARNING("Duplicate initializer[%s], the model will use the last initializer, please modify the model",
                        tensor->name().c_str());
        }
        // add. if the initializer has already existed in the graph, replace it.
        ir_graph->add_initializer(std::move(tensor));
    }

    // Step 4. Process the outputs name, type, shape information of the graph.
    for (auto& output : onnx_graph.output()) {
        if (!output.name().empty()) {
            const onnx::TypeProto& type = output.type();
            if (type.value_case() == onnx::TypeProto::ValueCase::kTensorType) {
                const onnx::TypeProto_Tensor tensor_type = type.tensor_type();
                PrimitiveDataType dt =
                    tensor_datatype_to_primitive(static_cast<onnx::TensorProto_DataType>(tensor_type.elem_type()));
                if (dt == PrimitiveDataType::UNKNOWN) {
                    return Status(StatusCode::INVALID_MODEL, "unsupported data type of graph outputs");
                }

                TensorShape shape = shapeproto_to_tensorshape(tensor_type.shape());
                NodeArg arg(output.name(), dt, shape);
                name_to_nodearg_map.emplace(output.name(), arg);
                ir_graph->get_or_create_nodearg(output.name(), arg);
                ir_graph->add_output_name(output.name());
            } else {
                LOG_WARNING("Graph output [%s] has not type or unsupported type", output.name().c_str());
            }
        } else {
            LOG_WARNING("Graph output name is empty");
        }
    }

    // Step 4. Process the ValueInfoProto name, type, shape information of the graph.
    for (auto& val_info : onnx_graph.value_info()) {
        if (!val_info.name().empty()) {
            const onnx::TypeProto& type = val_info.type();
            if (type.value_case() == onnx::TypeProto::ValueCase::kTensorType) {
                const onnx::TypeProto_Tensor tensor_type = type.tensor_type();
                PrimitiveDataType dt =
                    tensor_datatype_to_primitive(static_cast<onnx::TensorProto_DataType>(tensor_type.elem_type()));
                if (dt == PrimitiveDataType::UNKNOWN) {
                    return Status(StatusCode::INVALID_MODEL, "unsupported data type of graph value infos");
                }

                TensorShape shape = shapeproto_to_tensorshape(tensor_type.shape());
                NodeArg arg(val_info.name(), dt, shape);
                name_to_nodearg_map.emplace(val_info.name(), arg);
            }
        } else {
            LOG_WARNING("Graph value_info name is empty");
        }
    }

    // Step 5. Process the nodes in the graph
    int node_id = -1;
    for (auto& proto_node : onnx_graph.node()) {
        // skip the 'Constant' node
        if (proto_node.op_type() == "Constant") {
            continue;
        }
        ++node_id;
        std::unique_ptr<Node> ir_node;
        auto ret = parse_onnx_node(proto_node, ir_node, node_id, ir_graph.get(), name_to_nodearg_map);
        if (!ret.is_ok()) {
            return ret;
        }

        ir_graph->add_node(std::move(ir_node));
    }

    // Step 6. Initialize the state of this graph
    auto ret = ir_graph->initialize();
    return ret;
}

Status OnnxSerializer::parse_onnx_node(const onnx::NodeProto& onnx_node, std::unique_ptr<Node>& ir_node, int node_id,
                                       Graph* graph, const std::unordered_map<std::string, NodeArg>& nodearg_map) {
    auto create_node_args = [&](const std::vector<std::string>& names) {
        std::vector<NodeArg*> results;
        for (auto& name : names) {
            auto it = nodearg_map.find(name);
            if (it != nodearg_map.end()) {
                auto ret = graph->get_or_create_nodearg(name, it->second);
                results.emplace_back(ret);
            } else {
                auto ret = graph->get_or_create_nodearg(name, NodeArg(name));
                results.emplace_back(ret);
            }
        }
        return results;
    };

    std::vector<std::string> inputs_name;
    std::vector<std::string> outputs_name;

    for (auto& name : onnx_node.input()) {
        inputs_name.emplace_back(name);
    }

    for (auto& name : onnx_node.output()) {
        outputs_name.emplace_back(name);
    }

    auto node_input_args = create_node_args(inputs_name);
    auto node_output_args = create_node_args(outputs_name);
    auto name = onnx_node.name();
    auto type = onnx_node.op_type();
    auto doc_str = onnx_node.doc_string();
    auto domain = onnx_node.domain();

    std::unordered_map<std::string, std::unique_ptr<NodeAttribute>> attributes;
    for (auto& item : onnx_node.attribute()) {
        std::unique_ptr<NodeAttribute> attr;
        auto ret = parse_onnx_attribute(item, attr);
        if (!ret.is_ok()) {
            return ret;
        }

        attributes.emplace(item.name(), std::move(attr));
    }

    auto node = std::make_unique<Node>(node_id, *graph);
    node->init(name, type, domain, doc_str, node_input_args, node_output_args, std::move(attributes));
    ir_node = std::move(node);

    return Status::ok();
}

Status OnnxSerializer::retrieve_tensor_data(const onnx::TensorProto& proto_tensor, std::unique_ptr<Tensor>& ir_tensor,
                                            IAllocator* allocator, const std::string& name) {
    switch (proto_tensor.data_type()) {
        case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT: {
            auto tensor = std::make_unique<Tensor>(name);

            TensorShape tensor_shape;
            // add dims to the tensor
            int dim_size = proto_tensor.dims_size();
            for (int i = 0; i < dim_size; ++i) {
                int64_t dim = proto_tensor.dims(i);
                tensor_shape.add_dim(dim);
            }

            // initialize the ir tensor
            auto status = tensor->init(PrimitiveDataType::FLOAT32, tensor_shape, allocator);
            if (!status.is_ok()) {
                std::ostringstream oss;
                oss << "init tensor failed, tensor proto: " << proto_tensor.name();
                return Status(status.code(), oss.str());
            }

            if (proto_tensor.raw_data().length() > 0) {
                if (proto_tensor.raw_data().length() == sizeof(float) * tensor->shape().element_num()) {
                    const float* tensor_data = reinterpret_cast<const float*>(proto_tensor.raw_data().data());
                    float* ir_data = tensor->data_as<float>();
                    memmove(ir_data, tensor_data, sizeof(float) * tensor->shape().element_num());
                } else {
                    return Status(StatusCode::INVALID_MODEL, "Invalid tensor raw data length with its dims");
                }
            } else {
                if (proto_tensor.float_data_size() == tensor->shape().element_num()) {
                    float* ir_data = tensor->data_as<float>();
                    for (int i = 0; i < proto_tensor.float_data_size(); ++i) {
                        ir_data[i] = proto_tensor.float_data(i);
                    }
                } else {
                    return Status(StatusCode::INVALID_MODEL, "Invalid tensor float data length with its dims");
                }
            }

            ir_tensor = std::move(tensor);
            return Status::ok();
        }

        default: {
            std::ostringstream oss;
            oss << "not support data type for proto tensor";
            return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
        }
    }
    return Status::ok();
}

Status OnnxSerializer::constant_protonode_to_tensor(const onnx::NodeProto& proto_node,
                                                    std::unique_ptr<Tensor>& ir_tensor) {
    // Step 1. check the outputs size
    int output_size = proto_node.output_size();
    if (output_size == 0) {
        std::ostringstream oss;
        oss << "Constant node [" << proto_node.name() << "] has no outputs";
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    // Step 2. check the attributes size
    int attrs_size = proto_node.attribute_size();
    if (attrs_size == 0) {
        std::ostringstream oss;
        oss << "Constant node [" << proto_node.name() << "] has no attributes";
        return Status(StatusCode::INVALID_MODEL, oss.str());
    }

    const std::string& name = proto_node.output(0);
    const onnx::AttributeProto& const_attr = proto_node.attribute(0);

    IAllocator* allocator = AllocatorManager::instance()->get_allocator(IAllocator::Type::CPU);

    // Step 3. retrieve the data from the attribute
    switch (const_attr.type()) {
        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR: {
            const onnx::TensorProto& tensor = const_attr.t();
            return retrieve_tensor_data(tensor, ir_tensor, allocator, name);
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT: {
            float val = const_attr.f();
            ir_tensor = std::make_unique<Tensor>(name);
            TensorShape tensor_shape;
            tensor_shape.add_dim(1);
            auto status = ir_tensor->init(PrimitiveDataType::FLOAT32, tensor_shape, allocator);
            if (!status.is_ok()) {
                std::ostringstream oss;
                oss << "convert constant node float tensor failed, node proto: " << proto_node.name();
                return Status(status.code(), oss.str());
            }

            float* ir_data = ir_tensor->data_as<float>();
            ir_data[0] = val;

            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS: {
            TensorShape tensor_shape;
            tensor_shape.add_dim(const_attr.floats_size());

            ir_tensor = std::make_unique<Tensor>(name);
            auto status = ir_tensor->init(PrimitiveDataType::FLOAT32, tensor_shape, allocator);
            if (!status.is_ok()) {
                std::ostringstream oss;
                oss << "convert constant node floats tensor failed, node proto: " << proto_node.name();
                return Status(status.code(), oss.str());
            }

            float* ir_data = ir_tensor->data_as<float>();
            for (int i = 0; i < const_attr.floats_size(); ++i) {
                ir_data[i] = const_attr.floats(i);
            }

            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT: {
            int64_t val = const_attr.i();
            ir_tensor = std::make_unique<Tensor>(name);
            TensorShape tensor_shape;
            tensor_shape.add_dim(1);
            auto status = ir_tensor->init(PrimitiveDataType::INT64, tensor_shape, allocator);
            if (!status.is_ok()) {
                std::ostringstream oss;
                oss << "convert constant node int tensor failed, node proto: " << proto_node.name();
                return Status(status.code(), oss.str());
            }

            int64_t* ir_data = ir_tensor->data_as<int64_t>();
            ir_data[0] = val;

            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS: {
            TensorShape tensor_shape;
            tensor_shape.add_dim(const_attr.ints_size());

            ir_tensor = std::make_unique<Tensor>(name);
            auto status = ir_tensor->init(PrimitiveDataType::INT64, tensor_shape, allocator);
            if (!status.is_ok()) {
                std::ostringstream oss;
                oss << "convert constant node ints tensor failed, node proto: " << proto_node.name();
                return Status(status.code(), oss.str());
            }

            int64_t* ir_data = ir_tensor->data_as<int64_t>();
            for (int i = 0; i < const_attr.ints_size(); ++i) {
                ir_data[i] = const_attr.ints(i);
            }

            break;
        }

        default: {
            std::ostringstream oss;
            oss << "not supported attributes of constant node: " << proto_node.name();
            return Status(StatusCode::INVALID_MODEL, oss.str());

            break;
        }
    }

    return Status::ok();
}

PrimitiveDataType OnnxSerializer::tensor_datatype_to_primitive(const onnx::TensorProto_DataType& data_type) {
    PrimitiveDataType dt = PrimitiveDataType::UNKNOWN;

    if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_FLOAT) {
        dt = PrimitiveDataType::FLOAT32;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_INT8) {
        dt = PrimitiveDataType::INT8;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_UINT8) {
        dt = PrimitiveDataType::UINT8;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_INT16) {
        dt = PrimitiveDataType::INT16;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_UINT16) {
        dt = PrimitiveDataType::UINT16;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_INT32) {
        dt = PrimitiveDataType::INT32;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_UINT32) {
        dt = PrimitiveDataType::UINT32;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_INT64) {
        dt = PrimitiveDataType::INT64;
    } else if (data_type == onnx::TensorProto_DataType::TensorProto_DataType_UINT64) {
        dt = PrimitiveDataType::UINT64;
    }

    return dt;
}

TensorShape OnnxSerializer::shapeproto_to_tensorshape(const onnx::TensorShapeProto& shape_proto) {
    TensorShape shape;
    for (auto& dim : shape_proto.dim()) {
        if (dim.value_case() == onnx::TensorShapeProto_Dimension::ValueCase::kDimValue) {
            int64_t dim_val = dim.dim_value();
            shape.add_dim(dim_val);
        } else {
            shape.add_dim(-1);
        }
    }

    return shape;
}

NodeAttributeType OnnxSerializer::convert_to_node_attrtype(const onnx::AttributeProto_AttributeType& type) {
    NodeAttributeType dt = NodeAttributeType::INVALID;

    if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        dt = NodeAttributeType::INT64;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        dt = NodeAttributeType::FLOAT;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        dt = NodeAttributeType::STRING;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR) {
        dt = NodeAttributeType::TENSOR;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS) {
        dt = NodeAttributeType::INT64_ARRAY;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS) {
        dt = NodeAttributeType::FLOAT_ARRAY;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS) {
        dt = NodeAttributeType::STRING_ARRAY;
    } else if (type == onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS) {
        dt = NodeAttributeType::TENSOR_ARRAY;
    }

    return dt;
}

Status OnnxSerializer::parse_onnx_attribute(const onnx::AttributeProto& proto_attr,
                                            std::unique_ptr<NodeAttribute>& node_attr) {
    auto attr_type = convert_to_node_attrtype(proto_attr.type());
    if (attr_type == NodeAttributeType::INVALID) {
        return Status(StatusCode::INVALID_MODEL, "unsupport node attribute data type");
    }
    node_attr = std::make_unique<NodeAttribute>(proto_attr.name(), attr_type);

    switch (proto_attr.type()) {
        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT: {
            float val = proto_attr.f();
            node_attr->set_float(val);
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT: {
            int64_t val = proto_attr.i();
            node_attr->set_int64(val);
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING: {
            const std::string& val = proto_attr.s();
            node_attr->set_string(val);
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR: {
            const onnx::TensorProto& tensor = proto_attr.t();
            IAllocator* allocator = AllocatorManager::instance()->get_allocator(IAllocator::Type::CPU);
            std::unique_ptr<Tensor> ir_tensor;
            Status status = retrieve_tensor_data(tensor, ir_tensor, allocator, tensor.name());
            if (!status.is_ok()) {
                return status;
            }
            node_attr->set_tensor(std::move(ir_tensor));
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS: {
            for (auto& item : proto_attr.floats()) {
                node_attr->add_float(item);
            }
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS: {
            for (auto& item : proto_attr.ints()) {
                node_attr->add_int64(item);
            }
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS: {
            for (auto& item : proto_attr.strings()) {
                node_attr->add_string(item);
            }
            break;
        }

        case onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS: {
            for (auto& tensor : proto_attr.tensors()) {
                IAllocator* allocator = AllocatorManager::instance()->get_allocator(IAllocator::Type::CPU);
                std::unique_ptr<Tensor> ir_tensor;
                Status status = retrieve_tensor_data(tensor, ir_tensor, allocator, tensor.name());
                if (!status.is_ok()) {
                    return status;
                }
                node_attr->add_tensor(std::move(ir_tensor));
            }
            break;
        }

        default: {
            std::ostringstream oss;
            oss << "not supported attribute: " << proto_attr.name();
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }
    }

    return Status::ok();
}

}    // namespace io
}    // namespace simple_ai