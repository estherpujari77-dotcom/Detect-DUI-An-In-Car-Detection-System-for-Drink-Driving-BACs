from django.contrib import admin
from django.urls import path, re_path
from Remote_User import views as remoteuser
from Service_Provider import views as serviceprovider
from detectdui import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', remoteuser.index, name='index'),
    path('login/', remoteuser.login, name='login'),
    path('Register1/', remoteuser.Register1, name='Register1'),
    path('Predict_Drink_Driving_Detection/', remoteuser.Predict_Drink_Driving_Detection, name='Predict_Drink_Driving_Detection'),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name='ViewYourProfile'),
    path('logout/', remoteuser.logout_view, name='logout'),
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name='serviceproviderlogin'),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name='View_Remote_Users'),
    re_path(r'^charts/(?P<chart_type>\w+)', serviceprovider.charts, name='charts'),
    re_path(r'^charts1/(?P<chart_type>\w+)', serviceprovider.charts1, name='charts1'),
    re_path(r'^likeschart/(?P<like_chart>\w+)', serviceprovider.likeschart, name='likeschart'),
    path('View_Prediction_Of_Drink_Driving_Detection_Ratio/', serviceprovider.View_Prediction_Of_Drink_Driving_Detection_Ratio, name='View_Prediction_Of_Drink_Driving_Detection_Ratio'),
    path('train_model/', serviceprovider.train_model, name='train_model'),
    path('View_Prediction_Of_Drink_Driving_Detection/', serviceprovider.View_Prediction_Of_Drink_Driving_Detection, name='View_Prediction_Of_Drink_Driving_Detection'),
    path('Download_Predicted_DataSets/', serviceprovider.Download_Predicted_DataSets, name='Download_Predicted_DataSets'),
    path('sp_logout/', serviceprovider.sp_logout, name='sp_logout'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
