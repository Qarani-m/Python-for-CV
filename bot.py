import telegram.ext as telegram
import os
import pathlib
# turn sys off
# # turn apps off and on
# git 

# 
key ="5758276265:AAGZaVLrrGnYkz3T_t7C8A9ra329O5xzep4"

git = ["git add .", "git commit -m '...'","git push"]

def aux(update,context):
      message =update.message.text.split(" ")
      payload = message[0]
      if payload == "sysoff":
            os.system("shutdown /s /t 0")    
      if payload == "appoff":
            try:
              os.system(f"taskkill /F /IM {message[1]}")
            except Exception as e:
              update.message.reply_text("Something's wrong with appOff")
      if payload == "git":
            home_ = f"{pathlib.Path.home()}\\3d Objects\Lorr\{message[1]}"
            folder = message[1]
            try:
              reply = update.message.reply_text(os.system(f"cd home_ && {git[0]} && {git[1]} && {git[2]}"))
              
              match reply:
                case 0:
                  update.message.reply_text("[✔✔]")
                case 1:
                  update.message.reply_text("[❌❌]")
              
            except Exception as e:
              update.message.reply_text("Something's wrong with git")         
updater = telegram.Updater(key,use_context=True)
disp = updater.dispatcher
disp.add_handler(telegram.MessageHandler(telegram.Filters.text,aux))
updater.start_polling()
updater.idle()