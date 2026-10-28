for pkg in ['fastapi', 'uvicorn', 'flask', 'streamlit', 'cv2', 'torch', 'ultralytics']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"{pkg:15s}: {ver}")
    except ImportError:
        print(f"{pkg:15s}: NOT INSTALLED")
