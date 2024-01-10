#ifndef _H_SIMPLE_AI_IO_ONNX_SERIALIZER_H_
#define _H_SIMPLE_AI_IO_ONNX_SERIALIZER_H_

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "common/common.h"
#include "framework/allocator.h"
#include "framework/allocator_manager.h"
#include "ir/model.h"
#include "onnx.proto3.pb.h"
#include "utils/utils.h"

using namespace simple_ai::common;
using namespace simple_ai::ir;

namespace simple_ai {
namespace io {

/**
 * @brief Onnx proto file serializer
 *
 */
class OnnxSerializer final {
public:
    OnnxSerializer() = default;
    ~OnnxSerializer() = default;

    /**
     * @brief load from file path
     *
     * @param file_path the file path
     * @param model_ptr output parameter. the loaded model
     * @return Status
     */
    static Status load_from_file(const std::string& file_path, std::shared_ptr<Model>& model_ptr);

    /**
     * @brief load from memory
     *
     * @param data the data memory pointer
     * @param data_len the data length
     * @param model_ptr output parameter. the loaded model
     * @return Status
     */
    static Status load_from_memory(const void* data, size_t data_len, std::shared_ptr<Model>& model_ptr);

private:
    /**
     * @brief load onnx proto model
     *
     * @param loader the loader
     * @param model_ptr output parameter. the model ir
     * @return Status
     */
    static Status load_with_loader(std::function<Status(onnx::ModelProto&)> loader, std::shared_ptr<Model>& model_ptr);

    /**
     * @brief Validate the onnx proto model
     *
     * @param model the onnx proto model
     * @return Status
     */
    static Status validate_onnx_proto(const onnx::ModelProto& model);

    /**
     * @brief parse onnx model to ir model
     *
     * @param onnx_model the onnx model
     * @param ir_model output parameter. the ir model
     * @return Status
     */
    static Status parse_onnx_model(const onnx::ModelProto& onnx_model, std::shared_ptr<Model>& ir_model);

    /**
     * @brief parse onnx graph to ir graph
     *
     * @param onnx_graph the onnx graph
     * @param ir_graph output parameter. the ir graph
     * @return Status
     */
    static Status parse_onnx_graph(const onnx::GraphProto& onnx_graph, std::unique_ptr<Graph>& ir_graph);

    /**
     * @brief parse onnx node to ir node
     *
     * @param onnx_node the onnx node
     * @param ir_node output parameter. the ir node
     * @param node_id the ir node id
     * @param graph the graph which the ir node belongs to
     * @param nodearg_map the node args
     * @return Status
     */
    static Status parse_onnx_node(const onnx::NodeProto& onnx_node, std::unique_ptr<Node>& ir_node, int node_id,
                                  Graph* graph, const std::unordered_map<std::string, NodeArg>& nodearg_map);

    /**
     * @brief parse onnx node attribute to ir node attribute
     * 
     * @param proto_attr the onnx node attribute
     * @param node_attr the ir node attribute
     * @return Status 
     */
    static Status parse_onnx_attribute(const onnx::AttributeProto& proto_attr, std::unique_ptr<NodeAttribute>& node_attr);

    /**
     * @brief retrieve data from proto tensor, and save to ir tensor
     *
     * @param proto_tensor the proto tensor
     * @param ir_tensor output parameter. the ir tensor
     * @param allocator the allocator for ir tensor to allocate memory
     * @param name the name of the tensor
     * @return Status
     */
    static Status retrieve_tensor_data(const onnx::TensorProto& proto_tensor, std::unique_ptr<Tensor>& ir_tensor,
                                       IAllocator* allocator, const std::string& name);

    /**
     * @brief Convert 'Constant' proto node to tensor
     *
     * @param proto_node the proto node which op_type is 'Constant'
     * @param ir_tensor output parameter. the tensor ir
     * @return Status
     */
    static Status constant_protonode_to_tensor(const onnx::NodeProto& proto_node, std::unique_ptr<Tensor>& ir_tensor);

    /**
     * @brief convert proto tensor data type to primitive
     *
     * @param data_type the proto tensor data type
     * @return PrimitiveDataType
     */
    static PrimitiveDataType tensor_datatype_to_primitive(const onnx::TensorProto_DataType& data_type);

    /**
     * @brief convert proto shape to tensor shape
     *
     * @param shape_proto the proto shape
     * @return TensorShape
     */
    static TensorShape shapeproto_to_tensorshape(const onnx::TensorShapeProto& shape_proto);

    /**
     * @brief convert proto attribute type to node attributes
     *
     * @param type the proto attribute type
     * @return NodeAttributeType
     */
    static NodeAttributeType convert_to_node_attrtype(const onnx::AttributeProto_AttributeType& type);

private:
    SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxSerializer);
};

}    // namespace io
}    // namespace simple_ai

#endif