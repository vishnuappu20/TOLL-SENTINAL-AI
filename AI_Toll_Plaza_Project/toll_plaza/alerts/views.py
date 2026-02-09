from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect
from .models import AlertEvidence, UploadedVideo


def alert_dashboard(request):
    query = request.GET.get('q')

    if query:
        alerts = AlertEvidence.objects.filter(
            plate_number__icontains=query
        ).order_by('-detected_time')
    else:
        alerts = AlertEvidence.objects.all().order_by('-detected_time')

    return render(request, 'alerts/dashboard.html', {
        'alerts': alerts,
        'query': query
    })


@csrf_protect
def upload_video(request):
    if request.method == "POST":
        video = request.FILES.get("video")

        if video:
            UploadedVideo.objects.create(
                video=video,
                status="PENDING"
            )

        # 🔑 VERY IMPORTANT: redirect after POST
        return redirect("alert_dashboard")

    # ❗ If someone opens /upload/ via browser
    return redirect("alert_dashboard")
