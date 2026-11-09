#define OCR_LAYOUT_LIBRARY 1
#include "../../tools/ocr-ggml/doclayout-ggml.cpp"
#include "../../ggml/src/ggml-spacemit/ocr-im2col-pack.h"

// Static eligibility on the real, output-pruned graph. No inference, repacking
// or SpacemiT initialization; this is not a hardware coverage measurement.
int main(int argc,char ** argv) {
    try {
        require(argc==2,"usage: audit-layout-im2col-pack model.gguf");
        doclayout_graph model;model.native_elementwise=true;model.native_compact=true;
        model.load_layout(argv[1],4,{});
        json matches=json::array();uint64_t patches=0,sources=0;size_t max_scratch=0;
        int quantized_convolutions=0;
        for(int i=0;i<ggml_graph_n_nodes(model.graph);++i) {
            auto * node=ggml_graph_node(model.graph,i);
            if(node->op==GGML_OP_MUL_MAT&&ggml_is_quantized(node->src[0]->type)) {
                auto * source=node->src[1];
                while(source&&source->view_src)source=source->view_src;
                quantized_convolutions+=source&&source->op==GGML_OP_IM2COL;
            }
            const int matrix_index=ocr_im2col_pack::match(model.graph,i);
            if(matrix_index<0)continue;
            auto * matrix=ggml_graph_node(model.graph,matrix_index);
            const uint64_t patch_bytes=ggml_nbytes(node),image_bytes=ggml_nbytes(node->src[1]);
            const size_t scratch=ocr_im2col_pack::scratch_bytes(matrix,8);
            patches+=patch_bytes;sources+=image_bytes;max_scratch=std::max(max_scratch,scratch);
            matches.push_back({{"node_index",i},{"matrix_index",matrix_index},{"name",matrix->name},
                {"k",matrix->src[0]->ne[0]},{"n",matrix->ne[0]},{"m",matrix->ne[1]},
                {"weight_type",ggml_type_name(matrix->src[0]->type)},
                {"patch_bytes",patch_bytes},{"source_bytes",image_bytes},{"scratch_bytes_8_workers",scratch}});
        }
        require(matches.size()==size_t(quantized_convolutions),"quantized convolution chain not eligible");
        std::printf("%s\n",json{{"passed",true},{"matched_chains",matches.size()},
            {"quantized_convolution_nodes",quantized_convolutions},{"patch_bytes_static",patches},
            {"source_bytes_static",sources},{"max_scratch_bytes_8_workers",max_scratch},
            {"model_backend",model.backend_info()},{"matches",matches},
            {"compute_called",false},{"hardware_dispatch_verified",false},{"board_executed",false}}.dump().c_str());
        return 0;
    }catch(const std::exception & error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
