from detection.models import VehicleMaster
from alerts.models import AlertEvidence
from django.core.mail import send_mail
from django.conf import settings


def verify_vehicle(vehicle_type, plate, color, image=None, detected_vehicle=None):

    detected_plate = plate.upper()
    detected_color = color.capitalize()
    detected_type = vehicle_type.lower()

    try:
        vehicle = VehicleMaster.objects.get(
            plate_number=detected_plate
        )

        # ================= STOLEN VEHICLE =================
        if vehicle.status == "STOLEN":

            AlertEvidence.objects.create(
                detected_vehicle=detected_vehicle,
                plate_number=detected_plate,
                alert_reason="STOLEN VEHICLE",
                toll_id=detected_vehicle.toll_id,
                toll_name=detected_vehicle.toll_name,
                lane_number=detected_vehicle.lane_number,
                owner_name=vehicle.owner_name,
                owner_contact=vehicle.contact_info,
                image=image
            )

            send_mail(
                subject="🚨 Toll Plaza Alert - STOLEN VEHICLE",
                message=f"""
ALERT TYPE: STOLEN VEHICLE
PLATE: {detected_plate}
TOLL: {detected_vehicle.toll_name}
LANE: {detected_vehicle.lane_number}
OWNER: {vehicle.owner_name}
CONTACT: {vehicle.contact_info}
""",
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=["thejusprakashpersonal@gmail.com"],
                fail_silently=False,
            )

            return {"status": "ALERT", "reason": "STOLEN VEHICLE"}

        # ================= VEHICLE TYPE MISMATCH =================
        if vehicle.vehicle_type.lower() != detected_type:

            AlertEvidence.objects.create(
                detected_vehicle=detected_vehicle,
                plate_number=detected_plate,
                alert_reason="VEHICLE TYPE MISMATCH",
                toll_id=detected_vehicle.toll_id,
                toll_name=detected_vehicle.toll_name,
                lane_number=detected_vehicle.lane_number,
                owner_name=vehicle.owner_name,
                owner_contact=vehicle.contact_info,
                image=image
            )

            send_mail(
                subject="🚨 Toll Plaza Alert - TYPE MISMATCH",
                message=f"""
ALERT TYPE: VEHICLE TYPE MISMATCH
PLATE: {detected_plate}
DETECTED TYPE: {detected_type}
REGISTERED TYPE: {vehicle.vehicle_type}
TOLL: {detected_vehicle.toll_name}
LANE: {detected_vehicle.lane_number}
""",
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=["thejusprakashpersonal@gmail.com"],
                fail_silently=False,
            )

            return {"status": "ALERT", "reason": "TYPE MISMATCH"}

        # ================= COLOR MISMATCH =================
        if vehicle.registered_color.lower() != detected_color.lower():

            AlertEvidence.objects.create(
                detected_vehicle=detected_vehicle,
                plate_number=detected_plate,
                alert_reason="COLOR MISMATCH",
                toll_id=detected_vehicle.toll_id,
                toll_name=detected_vehicle.toll_name,
                lane_number=detected_vehicle.lane_number,
                owner_name=vehicle.owner_name,
                owner_contact=vehicle.contact_info,
                image=image
            )

            send_mail(
                subject="🚨 Toll Plaza Alert - COLOR MISMATCH",
                message=f"""
ALERT TYPE: COLOR MISMATCH
PLATE: {detected_plate}
DETECTED COLOR: {detected_color}
REGISTERED COLOR: {vehicle.registered_color}
TOLL: {detected_vehicle.toll_name}
LANE: {detected_vehicle.lane_number}
""",
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=["thejusprakashpersonal@gmail.com"],
                fail_silently=False,
            )

            return {"status": "ALERT", "reason": "COLOR MISMATCH"}

        # ================= NORMAL VEHICLE =================
        return {"status": "OK", "reason": "NORMAL VEHICLE"}

    except VehicleMaster.DoesNotExist:

        AlertEvidence.objects.create(
            detected_vehicle=detected_vehicle,
            plate_number=detected_plate,
            alert_reason="PLATE NOT FOUND",
            toll_id=detected_vehicle.toll_id,
            toll_name=detected_vehicle.toll_name,
            lane_number=detected_vehicle.lane_number,
            image=image
        )

        send_mail(
            subject="🚨 Toll Plaza Alert - UNKNOWN VEHICLE",
            message=f"""
ALERT TYPE: PLATE NOT FOUND
PLATE: {detected_plate}
TOLL: {detected_vehicle.toll_name}
LANE: {detected_vehicle.lane_number}
""",
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=["thejusprakashpersonal@gmail.com"],
            fail_silently=False,
        )

        return {"status": "ALERT", "reason": "PLATE NOT FOUND"}