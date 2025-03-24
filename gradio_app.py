import gradio as gr
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model.emotion_cnn import EmotionCNN

# Load model
device = torch.device("cpu")
model = EmotionCNN().to(device)
checkpoint = torch.load("model/emotion_cnn_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define class labels
class_mapping = ['Angry', 'Fear', 'Happy', 'Sad', 'Neutral']

# Define song recommendations for each emotion with YouTube video links and playlist
song_recommendations = {
    'Angry': [
        ("🎵 Energetic Release Playlist - Perfect for letting out steam", "https://youtube.com/playlist?list=PLWSvYmatuZkhOpJy5nKtgFL17Ods4wDKD"),
        ("⚡ Taylor Swift - Look What You Made Me Do", "https://www.youtube.com/watch?v=3tmd-ClpJxA"),
        ("🔥 Taylor Swift - Bad Blood", "https://www.youtube.com/watch?v=QcIy9NiNbmo"),
        ("💪 (G)I-DLE - TOMBOY", "https://www.youtube.com/watch?v=z0zP5r5NlCg"),
        ("💥 (G)I-DLE - Nxde", "https://www.youtube.com/watch?v=6t7KOQhLXRQ"),
        ("🎸 Linkin Park - In The End", "https://www.youtube.com/watch?v=eVTXPUF4Oz4"),
        ("🤘 BLACKPINK - Kill This Love", "https://www.youtube.com/watch?v=2S24-y0Ij3Y"),
        ("💢 Sai Sai - A Chit Chin Par", "https://www.youtube.com/watch?v=8EZh1GxzB8g"),
        ("⚔️ BTS - UGH!", "https://www.youtube.com/watch?v=1yxEmmYQdl8"),
        ("🔪 Eminem - Lose Yourself", "https://www.youtube.com/watch?v=_Yhyp-_hX2s")
    ],
    'Fear': [
        ("🎵 Calming Anxiety Playlist - Soothing melodies to ease your mind", "https://youtube.com/playlist?list=PLWSvYmatuZkhOpJy5nKtgFL17Ods4wDKD"),
        ("🌙 Taylor Swift - Anti-Hero", "https://www.youtube.com/watch?v=b1kbLwvqugk"),
        ("🌳 Taylor Swift - Out of the Woods", "https://www.youtube.com/watch?v=QcIy9NiNbmo"),
        ("🌌 (G)I-DLE - Villain Dies", "https://www.youtube.com/watch?v=JQGRg8XBnB4"),
        ("🖤 (G)I-DLE - Dark (X-file)", "https://www.youtube.com/watch?v=9kaCAbIXuyg"),
        ("🌊 IU - Through the Night", "https://www.youtube.com/watch?v=BzYnNdJhZQw"),
        ("🎭 Imagine Dragons - Demons", "https://www.youtube.com/watch?v=mWRsgZuwf_8"),
        ("🌿 Htoo Eain Thin - Min Mha Yay Ei Chin", "https://www.youtube.com/watch?v=7PXoD25hR8E"),
        ("🌙 TWICE - Feel Special", "https://www.youtube.com/watch?v=3ymwOvzhwHs"),
        ("💫 Sia - Breathe Me", "https://www.youtube.com/watch?v=ghPcYqn0p4Y")
    ],
    'Happy': [
        ("🎵 Feel-Good Vibes Playlist - Upbeat tunes to keep you smiling", "https://youtube.com/playlist?list=PLWSvYmatuZkhOpJy5nKtgFL17Ods4wDKD"),
        ("✨ Taylor Swift - Shake It Off", "https://www.youtube.com/watch?v=nfWlot6h_JM"),
        ("🎉 Taylor Swift - 22", "https://www.youtube.com/watch?v=AgFeZr5ptV8"),
        ("👑 (G)I-DLE - Queencard", "https://www.youtube.com/watch?v=Y8JFxS1HlDo"),
        ("💎 (G)I-DLE - My Bag", "https://www.youtube.com/watch?v=1qYz7rfgLWE"),
        ("🌟 Pharrell Williams - Happy", "https://www.youtube.com/watch?v=ZbZSe6N_BXs"),
        ("🎊 Big Bag - Bar Lar Nae", "https://www.youtube.com/watch?v=PF8HE8t6Qno"),
        ("💫 BTS - Dynamite", "https://www.youtube.com/watch?v=gdZLi9oWNZg"),
        ("🌈 TWICE - What is Love?", "https://www.youtube.com/watch?v=i0p1bmr0EmE"),
        ("🎪 Justin Timberlake - Can't Stop the Feeling!", "https://www.youtube.com/watch?v=ru0K8uYEZWw")
    ],
    'Sad': [
        ("🎵 Emotional Healing Playlist - Songs to help process your feelings", "https://youtube.com/playlist?list=PLWSvYmatuZkhluzeZGwQmcDWyktxinB7h"),
        ("💔 Taylor Swift - All Too Well (10 Minute Version)", "https://www.youtube.com/watch?v=3tmd-ClpJxA"),
        ("🌧️ Taylor Swift - Cardigan", "https://www.youtube.com/watch?v=K-a8s8OLBSE"),
        ("🌸 (G)I-DLE - Dahlia", "https://www.youtube.com/watch?v=0WxYwKp0XcA"),
        ("💫 (G)I-DLE - Already", "https://www.youtube.com/watch?v=6ZUIwj3FgUY"),
        ("🌊 Sai Sai - Ma Nae Nae Bu", "https://www.youtube.com/watch?v=8EZh1GxzB8g"),
        ("🥀 Adele - Someone Like You", "https://www.youtube.com/watch?v=hLQl3WQQoQ0"),
        ("💔 IU - Love Poem", "https://www.youtube.com/watch?v=kcx0a2OAhN0"),
        ("🌙 BIGBANG - Last Dance", "https://www.youtube.com/watch?v=--zku6TB5NY"),
        ("🌧️ Lewis Capaldi - Someone You Loved", "https://www.youtube.com/watch?v=zABLecsR5UE")
    ],
    'Neutral': [
        ("🎵 Mood Boost Playlist - Discover new favorites", "https://youtube.com/playlist?list=PLWSvYmatuZkhOpJy5nKtgFL17Ods4wDKD"),
        ("☀️ Taylor Swift - Cruel Summer", "https://www.youtube.com/watch?v=ic8j13piAhQ"),
        ("⭐ Taylor Swift - Karma", "https://www.youtube.com/watch?v=rg6t2fR5_90"),
        ("💫 (G)I-DLE - I DO", "https://www.youtube.com/watch?v=f5_QQ8xlwQg"),
        ("🌈 (G)I-DLE - Paradise", "https://www.youtube.com/watch?v=Y8JFxS1HlDo"),
        ("🎧 Lay Phyu - A Chit Tae A Chit", "https://www.youtube.com/watch?v=8EZh1GxzB8g"),
        ("🌟 Ed Sheeran - Perfect", "https://www.youtube.com/watch?v=2Vv-BfVoq4g"),
        ("✨ IU - Eight (feat. Suga)", "https://www.youtube.com/watch?v=TgOu00Mf3kI"),
        ("🎵 Red Velvet - Feel My Rhythm", "https://www.youtube.com/watch?v=R9At2ICm4LQ"),
        ("🎪 Coldplay - Yellow", "https://www.youtube.com/watch?v=yKNxeF4KMsY")
    ]
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def detect_emotion(image):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(50, 50))
    
    # Create a copy for drawing
    output_image = image.copy()
    
    result_text = ""
    recommendations_text = ""
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face_pil = Image.fromarray(face)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, 0)
            
        # Get emotion label and confidence
        emotion = class_mapping[predicted_class.item()]
        confidence_score = confidence.item() * 100
        
        # Draw on image
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion} ({confidence_score:.1f}%)"
        cv2.putText(output_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add to result text
        result_text += f"Detected emotion: {emotion} (Confidence: {confidence_score:.1f}%)\n"
        
        # Add song recommendations with links
        recommendations_text += f"\n<div style='background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-top: 10px;'>"
        recommendations_text += f"<h3 style='margin: 0 0 10px 0;'>🎵 Music Recommendations for {emotion} Mood</h3>"
        
        # First add the playlist as a special section
        playlist_title, playlist_link = song_recommendations[emotion][0]
        autoplay_link = f"{playlist_link}&shuffle=1"
        recommendations_text += f"<div style='margin-bottom: 15px;'>"
        recommendations_text += f"<strong>Featured Playlist:</strong><br>"
        recommendations_text += f"<a href='{autoplay_link}' target='_blank' style='color: #2196F3; text-decoration: none;'>{playlist_title}</a>"
        recommendations_text += "</div>"
        
        # Then add individual songs
        recommendations_text += "<strong>Recommended Songs:</strong><br>"
        for song_title, song_link in song_recommendations[emotion][1:]:
            autoplay_link = f"{song_link}&autoplay=1"
            recommendations_text += f"<a href='{autoplay_link}' target='_blank' style='color: #2196F3; text-decoration: none; display: block; margin: 5px 0;'>{song_title}</a>"
        
        recommendations_text += "<p style='margin: 10px 0 0 0; font-size: 0.9em; color: #666;'>💡 Click any link to open in YouTube. Songs will autoplay when opened.</p>"
        recommendations_text += "</div>"
    
    if not len(faces):
        result_text = "No face detected"
        recommendations_text = ""
    
    combined_text = result_text + recommendations_text
    return output_image, combined_text

# Create Gradio interface
iface = gr.Interface(
    fn=detect_emotion,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs=[
        gr.Image(label="Processed Image"),
        gr.HTML(label="Detection Results & Song Recommendations")
    ],
    live=True,
    title="Real-time Emotion Detection with Music Recommendations 🎵",
    description="Detect emotions in real-time using your webcam and get personalized song recommendations based on your mood! Each recommendation includes a curated playlist and individual songs. Simply click on any title to listen on YouTube! 🎧",
    theme="default"
)

# Launch the interface
iface.launch()