from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):

    ROLE_CHOICES = [
        ('ADMIN', 'Admin'),
        ('OPERATOR', 'Operator'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    toll_id = models.CharField(max_length=50, null=True, blank=True)
    toll_name = models.CharField(max_length=100, null=True, blank=True)
    lane_number = models.CharField(max_length=20, null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.role}"
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        if instance.is_superuser:
            UserProfile.objects.create(
                user=instance,
                role='ADMIN'
            )
        else:
            UserProfile.objects.create(
                user=instance,
                role='OPERATOR'
            )