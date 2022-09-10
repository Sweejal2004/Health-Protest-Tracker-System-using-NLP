from rest_framework import serializers
from .models import *


class ConvertSerializer(serializers.ModelSerializer):
    
    class Meta:
        model = protest_data
        fields = '__all__'

