from detection.models import VehicleMaster
from alerts.models import AlertEvidence
from django.core.mail import send_mail
from django.conf import settings

def verify_vehicle(vehicle_type, plate, color, image=None):
    detected_plate = plate.upper()
    detected_color = color.capitalize()
    detected_type = vehicle_type.lower()

    try:
        vehicle = VehicleMaster.objects.get(
            plate_number=detected_plate
        )

        # ========== STOLEN VEHICLE ==========
        if vehicle.status == "STOLEN":
            AlertEvidence.objects.create(
                plate_number=detected_plate,
                detected_color=detected_color,
                database_color=vehicle.registered_color,
                detected_vehicle_type=detected_type,
                database_vehicle_type=vehicle.vehicle_type,
                alert_type="STOLEN",
                image=image
            )
            alert_type = "STOLEN vehicle"

            send_mail(
             subject="🚨 Toll Plaza Alert",
             message=f"""
            ALERT TYPE: {alert_type}
            PLATE: {detected_plate}
            COLOR: {detected_color}
            VEHICLE TYPE: {detected_type}
            """,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=["thejusprakashpersonal@gmail.com"],
            fail_silently=False,
            )       
            return {
                "status": "ALERT",
                "reason": "STOLEN VEHICLE",
                "plate": detected_plate
            }
        # ========== VEHICLE TYPE MISMATCH ==========
        if vehicle.vehicle_type.lower() != detected_type:
            AlertEvidence.objects.create(
                plate_number=detected_plate,
                detected_color=detected_color,
                database_color=vehicle.registered_color,
                detected_vehicle_type=detected_type,
                database_vehicle_type=vehicle.vehicle_type,
                alert_type="FAKE_PLATE_TYPE",
                image=image
            )
            alert_type = "fake numberr plate-vehicle type mismatch"

            send_mail(
             subject="🚨 Toll Plaza Alert",
             message=f"""
            ALERT TYPE: {alert_type}
            PLATE: {detected_plate}
            COLOR: {detected_color}
            VEHICLE TYPE: {detected_type}
            """,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=["thejusprakashpersonal@gmail.com"],
            fail_silently=False,
            )
            return {
                "status": "ALERT",
                "reason": "VEHICLE TYPE MISMATCH",
                "plate": detected_plate
            }
        # ========== COLOR MISMATCH ==========
        if vehicle.registered_color.lower() != detected_color.lower():
            AlertEvidence.objects.create(
                plate_number=detected_plate,
                detected_color=detected_color,
                database_color=vehicle.registered_color,
                detected_vehicle_type=detected_type,
                database_vehicle_type=vehicle.vehicle_type,
                alert_type="FAKE_PLATE_COLOR",
                image=image
            )
            alert_type = "fake plate vehicle-colour mismatch"

            send_mail(
             subject="🚨 Toll Plaza Alert",
             message=f"""
            ALERT TYPE: {alert_type}
            PLATE: {detected_plate}
            COLOR: {detected_color}
            VEHICLE TYPE: {detected_type}
            """,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=["thejusprakashpersonal@gmail.com"],
            fail_silently=False,
            )
            return {
                "status": "ALERT",
                "reason": "COLOR MISMATCH",
                "plate": detected_plate
            }



        # ========== NORMAL ==========
        return {
            "status": "OK",
            "reason": "NORMAL VEHICLE",
            "plate": detected_plate
        }

    except VehicleMaster.DoesNotExist:
        # ========== PLATE NOT FOUND ==========
        AlertEvidence.objects.create(
            plate_number=detected_plate,
            detected_color=detected_color,
            detected_vehicle_type=detected_type,
            alert_type="UNKNOWN",
            image=image
        )
        alert_type = "plate not found" 
        send_mail(
         subject="🚨 Toll Plaza Alert",
         message=f"""
            ALERT TYPE: {alert_type}
            PLATE: {detected_plate}
            COLOR: {detected_color}
            VEHICLE TYPE: {detected_type}
            """,
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=["thejusprakashpersonal@gmail.com"],
            fail_silently=False,
            )
        return {
            "status": "ALERT",
            "reason": "PLATE NOT FOUND",
            "plate": detected_plate
        }
