// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Collections.Generic;

public class ImageData
{
    public string ImagePath { get; set; }
}

public class ImagePrediction
{
    [ColumnName("dense")]
    public float[] Score { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var context = new MLContext();

        // Load gambar contoh
        var samples = new List<ImageData>
        {
            new ImageData { ImagePath = "sample.jpg" } // ubah ke gambar kamu
        };

        var data = context.Data.LoadFromEnumerable(samples);

        // Pipeline pemrosesan dan model TensorFlow
        var pipeline = context.Transforms.LoadImages("input", "", nameof(ImageData.ImagePath))
            .Append(context.Transforms.ResizeImages("input", 224, 224))
            .Append(context.Transforms.ExtractPixels("input"))
            .Append(context.Model.LoadTensorFlowModel("model/model_deteksi_sampah.h5")
                .ScoreTensorName("dense"));

        var model = pipeline.Fit(data);
        var predictions = model.Transform(data);

        var result = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false);
        foreach (var pred in result)
        {
            Console.WriteLine($"Prediksi skor: {string.Join(", ", pred.Score)}");
        }
    }
}

