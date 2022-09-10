from .models import protest_data
from django import forms
from django.forms.widgets import NumberInput

class MyForm(forms.Form):
        d_date=forms.DateTimeField(label="", required=True, 
        widget=NumberInput(attrs={'type':'date','class': 'form-control', }))
       