from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from detection.utils import verify_vehicle


@csrf_exempt
def verify_vehicle_api(request):
    """
    API endpoint called by AI service.
    Expects:
      - plate (string)
      - color (string)
      - image (optional file)

    Returns:
      JSON with verification status
    """

    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST method allowed"},
            status=405
        )

    # Read data sent by AI (multipart/form-data)
    vehicle_type = request.POST.get("vehicle_type")
    plate = request.POST.get("plate")
    color = request.POST.get("color")
    
    image = request.FILES.get("image")  # optional

    # Validate input
    if not plate or not color or not vehicle_type:
      return JsonResponse({"error": "plate, color, vehicle_type required"}, status=400)


    try:
        # Call your existing verification logic
        result = verify_vehicle(vehicle_type, plate, color, image)

        # Wrap string result into JSON
        return JsonResponse({
            "plate": plate,
            "detected_color": color,
            "status": result
        })

    except Exception as e:
        # Any unexpected backend error
        return JsonResponse(
            {"error": str(e)},
            status=500
        )
