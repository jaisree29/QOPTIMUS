from django.shortcuts import render, redirect, reverse
from django.contrib import messages
from validate_email import validate_email
from .models import Profile
from .forms import LoginForm, SignUpForm
from .decorators import auth_user_should_not_access
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth import get_user_model

User = get_user_model()

@auth_user_should_not_access
def Login(request):
    form = SignUpForm()

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if not user:
            messages.error(request, '⚠️ Invalid credentials, try again')
            return render(request, 'Login.html')

        login(request, user)
        return redirect(reverse('Dashboard'))

    return render(request, 'Login.html', {'form': form})


@auth_user_should_not_access
def Register(request):
    form = SignUpForm()

    if request.method == "POST":
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if len(password1) < 6:
            messages.error(request, '⚠️ Password should be at least 6 characters')
            return redirect('Register')

        if password1 != password2:
            messages.error(request, '⚠️ Password Mismatch!')
            return redirect('Register')

        if not validate_email(email):
            messages.error(request, '⚠️ Invalid Email Address')
            return redirect('Register')

        if not username:
            messages.error(request, '⚠️ Username is required!')
            return redirect('Register')

        if User.objects.filter(username=username).exists():
            messages.error(request, '⚠️ Username is already taken!')
            return redirect('Register')

        if User.objects.filter(email=email).exists():
            messages.error(request, '⚠️ Email is already taken!')
            return redirect('Register')

        user = User.objects.create_user(
            first_name=first_name,
            last_name=last_name,
            username=username,
            email=email
        )
        user.set_password(password1)
        user.save()

        messages.success(request, '✅ Registration successful! You can now login.')
        return redirect('Login')

    return render(request, 'Register.html', {'form': form})


def Logout(request):
    logout(request)
    messages.success(request, '✅ Successfully Logged Out!')
    return redirect(reverse('Login'))


def Dashboard(request):
    return render(request, 'Dashboard.html')
