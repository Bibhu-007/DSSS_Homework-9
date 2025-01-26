import torch
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import pipeline

API_TOKEN = 'xxxxxxxxxxxxxxxxxxxxx'

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Hello! I am your AI Assistant. How can I help you today?")

def setup_tiny_llama():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    pipe = pipeline("text-generation", 
                    model=model_name, 
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, 
                    device=0 if device.type == "cuda" else -1)  
    return pipe

def handle_message(update: Update, context: CallbackContext):
    user_message = update.message.text
    print(f"Message received: {user_message}")
    
    
    pipe = setup_tiny_llama()
    response = pipe(user_message, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    
    update.message.reply_text(f"{response}")

def main():
    updater = Updater(API_TOKEN)
    dispatcher = updater.dispatcher
    
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    main()
