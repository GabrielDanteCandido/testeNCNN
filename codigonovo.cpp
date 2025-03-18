#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net mobilenet;

    mobilenet.opt.use_vulkan_compute = true;

    // Modelo convertido de https://github.com/chuanqi305/MobileNet-SSD
    // e pode ser baixado de https://drive.google.com/open?id=0ByaKLD9QaPtucWk0Y0dha1VVY0U
    // O modelo ncnn está em https://github.com/nihui/ncnn-assets/tree/master/models
    if (mobilenet.load_param("mobilenet_ssd_voc_ncnn.param"))
        exit(-1);
    if (mobilenet.load_model("mobilenet_ssd_voc_ncnn.bin"))
        exit(-1);

    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("detection_out", out);

    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}

static void print_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "Objeto detectado: %s (%.1f%%) em [x=%.2f, y=%.2f, largura=%.2f, altura=%.2f]\n",
                class_names[obj.label], obj.prob * 100,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Uso: %s [caminho_da_imagem]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 1);
    if (m.empty())
    {
        fprintf(stderr, "Erro ao carregar a imagem: %s\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_mobilenet(m, objects);

    // Em vez de exibir a imagem, apenas imprima os resultados
    print_objects(m, objects);

    return 0;
}
