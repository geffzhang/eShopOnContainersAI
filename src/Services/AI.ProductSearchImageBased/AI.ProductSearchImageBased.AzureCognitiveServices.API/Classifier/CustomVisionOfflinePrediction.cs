using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Transforms;
using SixLabors.Primitives;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorFlow;

namespace Microsoft.eShopOnContainers.Services.AI.ProductSearchImageBased.AzureCognitiveServices.API.Classifier
{
    public class TensorFlowNeuralNetworkSettings
    {
        public string InputTensorName { get; internal set; }
        public string OutputTensorName { get; internal set; }
        public string ModelFilename { get; internal set; }
        public string LabelsFilename { get; internal set; }
        public float Threshold { get; internal set; }
        public int InputTensorWidth { get; internal set; }
        public int InputTensorHeight { get; internal set; }
        public int InputTensorChannels { get; internal set; }
    }

    public class CustomVisionOfflinePrediction : IClassifier
    {
        protected TensorFlowNeuralNetworkSettings modelSettings;

        public static readonly TensorFlowNeuralNetworkSettings Settings = new TensorFlowNeuralNetworkSettings()
        {
            InputTensorName = "Placeholder",
            OutputTensorName = "loss",
            InputTensorWidth = 227,
            InputTensorHeight = 227,
            InputTensorChannels = 3,
            ModelFilename = "model.pb",
            LabelsFilename = "labels.txt",
            Threshold = 0.9f
        };

        private readonly AppSettings settings;
        private readonly IHostingEnvironment environment;
        private readonly ILogger<CustomVisionOfflinePrediction> logger;

        public CustomVisionOfflinePrediction(IOptionsSnapshot<AppSettings> settings, IHostingEnvironment environment, ILogger<CustomVisionOfflinePrediction> logger)
        {
            this.settings = settings.Value;
            this.environment = environment;
            this.logger = logger;
            modelSettings = Settings;
        }

        /// <summary>
        /// Classifiy Image using Deep Neural Networks
        /// </summary>
        /// <param name="image">image (jpeg) file to be analyzed</param>
        /// <returns>labels related to the image</returns>
        public Task<IEnumerable<LabelConfidence>> ClassifyImageAsync(byte[] image)
        {
            // TODO: new Task
            return Task.FromResult(Process(image, modelSettings));
        }

        private IEnumerable<LabelConfidence> Process(byte[] image, TensorFlowNeuralNetworkSettings settings)
        {
            var (model, labels) = LoadModelAndLabels(settings.ModelFilename, settings.LabelsFilename);
            var imageTensor = LoadImage(image);

            var result = Eval(model, imageTensor, settings.InputTensorName, settings.OutputTensorName, labels).ToArray();

            IEnumerable<LabelConfidence> labelsToReturn = result                            
                                    .Where(c => c.Probability >= settings.Threshold)
                                    .OrderByDescending(c => c.Probability);
            return labelsToReturn;
        }

        private TFTensor LoadImage(byte[] image)
        {
            using (var ms = new MemoryStream(image))
            {
                using (var imgRgb = Image.Load<Rgb24>(ms))
                {
                    logger.LogInformation($"Image info: width={imgRgb.Width}, height:{imgRgb.Height}");

                    var img = ResizeDownTo1600(imgRgb);

                    var minDim = Math.Min(img.Width, img.Height);
                    img = CropCenter(img, minDim, minDim);

                    img = ResizeTo256(img);

                    img = CropCenter(img, Settings.InputTensorWidth, Settings.InputTensorHeight);

                    SaveTemporalImage(img);

                    return ConvertToTensor(img);
                }
            }
        }

        private void SaveTemporalImage(Image<Rgb24> image)
        {
            using (var ms = new MemoryStream())
            {
                image.SaveAsJpeg(ms);
                var tempFileName = Path.Combine(Path.GetTempPath(), Path.ChangeExtension(Path.GetTempFileName(), "jpg"));
                logger.LogInformation($"Writing pre-processed image file at {tempFileName}");
                File.WriteAllBytes(tempFileName, ms.ToArray());
            }
        }

        private TFTensor ConvertToTensor(Image<Rgb24> image)
        {
            float[,,,] output = new float[1, Settings.InputTensorWidth, Settings.InputTensorHeight, Settings.InputTensorChannels];
            for (int row = 0; row < Settings.InputTensorWidth; row++)
            {
                for (int column = 0; column < Settings.InputTensorHeight; column++)
                {
                    var pixel = image[row, column];
                    output[0, row, column, 0] = pixel.B;
                    output[0, row, column, 1] = pixel.G;
                    output[0, row, column, 2] = pixel.R;
                }
            }
            return new TFTensor(output);
        }

        private Image<Rgb24> ResizeDownTo1600(Image<Rgb24> image)
        {
            const int maxSize = 1600;
            int height = image.Height, 
                width = image.Width;

            if (height < maxSize && width < maxSize)
                return image;

            int newHeight, newWidth;
            if (height > width)
                (newWidth, newHeight) = ((int)maxSize * width / height, maxSize);
            else
                (newWidth, newHeight) = (maxSize, (int)maxSize * height / width);
            image.Mutate(x => x.Resize(newWidth, newHeight, KnownResamplers.Triangle));
            return image;
        }

        private Image<Rgb24> ResizeTo256(Image<Rgb24> image)
        {
            const int maxSize = 256;
            image.Mutate(x => x.Resize(maxSize, maxSize, KnownResamplers.Triangle));
            return image;
        }

        private Image<Rgb24> CropCenter(Image<Rgb24> image, int x, int y)
        {
            int width = image.Width,
                height = image.Height;

            int sx = width / 2 - (x / 2);
            int sy = height / 2 - (y / 2);
            image.Mutate(i => i.Crop(new Rectangle(sx,sy,x,y)));
            return image;
        }

        private (TFGraph, string[]) LoadModelAndLabels(string modelFilename, string labelsFilename)
        {
            const string EmptyGraphModelPrefix = "";

            var modelsFolder = Path.Combine(environment.ContentRootPath, settings.AIModelsPath);

            modelFilename = Path.Combine(modelsFolder, modelFilename);
            if (!File.Exists(modelFilename))
                throw new ArgumentException("Model file not exists", nameof(modelFilename));

            var model = new TFGraph();
            model.Import(File.ReadAllBytes(modelFilename), EmptyGraphModelPrefix);

            labelsFilename = Path.Combine(modelsFolder, labelsFilename);
            if (!File.Exists(labelsFilename))
                throw new ArgumentException("Labels file not exists", nameof(labelsFilename));

            var labels = File.ReadAllLines(labelsFilename);

            return (model, labels);
        }

        private IEnumerable<LabelConfidence> Eval(TFGraph graph, TFTensor imageTensor, string inputTensorName, string outputTensorName, string[] labels)
        {
            using (var session = new TFSession(graph))
            {
                var runner = session.GetRunner();

                // Create an input layer to feed (tensor) image, 
                // fetch label in output layer
                var input = graph[inputTensorName][0];

                var output = graph[outputTensorName][0];

                runner.AddInput(input, imageTensor)
                      .Fetch(output);

                var results = runner.Run();

                // convert output tensor in float array
                var probabilities = (float[,])results[0].GetValue(jagged: false);

                var idx = 0;
                return from label in labels
                       select new LabelConfidence { Label = label, Probability = probabilities[0, idx++] };
            }
        }
    }
}
