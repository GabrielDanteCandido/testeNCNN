#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp> // Para exibir vídeos
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
        fprintf(stderr, "Uso: %s [caminho_do_video]\n", argv[0]);
        return -1;
    }

    const char* videopath = argv[1];

    // Abre o vídeo
    cv::VideoCapture cap(videopath);
    if (!cap.isOpened())
    {
        fprintf(stderr, "Erro ao abrir o vídeo: %s\n", videopath);
        return -1;
    }

    cv::Mat frame;
    while (true)
    {
        // Captura um frame do vídeo
        cap >> frame;
        if (frame.empty())
            break; // Sai do loop se o vídeo terminar

        // Detecta objetos no frame
        std::vector<Object> objects;
        detect_mobilenet(frame, objects);

        // Exibe os resultados no console
        print_objects(frame, objects);

        // (Opcional) Exibe o frame com as detecções
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];
            cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0), 2); // Desenha retângulo
            cv::putText(frame, class_names[obj.label], cv::Point(obj.rect.x, obj.rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2); // Rótulo
        }

        cv::imshow("Video", frame); // Exibe o frame
        if (cv::waitKey(1) == 27) // Aguarda 1 ms e verifica se a tecla ESC foi pressionada
            break;
    }

    cap.release(); // Libera o vídeo
    cv::destroyAllWindows(); // Fecha todas as janelas

    return 0;
}
