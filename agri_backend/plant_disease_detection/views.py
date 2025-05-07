import os
import logging
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from .serializers import ImageUploadSerializer
from .model_utils import load_model, predict, load_class_names

# Load the model and class names
model_path = os.path.join(settings.BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "model_weights.pth")
class_file = os.path.join(settings.BASE_DIR, "plant_disease_detection", "plant_disease_dataset", "class.txt")
num_classes = 38  # Replace with the actual number of classes
model = load_model(model_path, num_classes)
class_names = load_class_names(class_file)

logger = logging.getLogger(__name__)

class PredictDiseaseView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            # Ensure the media directory exists
            media_dir = os.path.join(settings.BASE_DIR, "media")
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)
                logger.info(f"Created media directory: {media_dir}")

            # Save the uploaded image
            image_file = serializer.validated_data['image']
            file_path = os.path.join(media_dir, image_file.name)
            logger.info(f"Saving file to: {file_path}")

            with open(file_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # Predict the disease
            predicted_class = predict(file_path, model, class_names)
            logger.info(f"Predicted class: {predicted_class}")

            # Return the prediction
            return Response({'prediction': predicted_class})
        logger.error(f"Serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=400)