
from django.contrib import admin
from django.urls import path , include
from users.views import home


urlpatterns = [
  path('', home, name='home'),
    path('admin/', admin.site.urls),
    path('api/user/', include('users.urls')),
    path('api/insect/', include('insect_model.urls')),
    path('api/ollama/', include('ollama_integration.urls')),
    path('api/land/', include('land_details.urls')),  # Include land_details app URLs
    path('api/plant/', include('plant_disease_detection.urls')),
]
