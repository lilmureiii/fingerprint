
import customtkinter

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("1000x500")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.geometry("1000x500")

        self.navigation = NavigationFrame(self, controller=self)
        self.navigation.grid(row=0, column=0, sticky="nesw")
        self.frame2 = Frame2(self, controller=self)
        self.frame1 = Frame1(self, controller=self)
        self.frame1.grid(row=0, column=1, sticky="nsew")
        self.frame2.grid(row=0, column=1, sticky="nsew")

class NavigationFrame(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(fg_color="grey15")

        title = customtkinter.CTkLabel(self, 
                                       text="Navigation", 
                                       anchor="center", 
                                       font=("Helvetica", 20))
        title.grid(row=0, column=0, padx=(20, 20))

        test1 = customtkinter.CTkButton(self, 
                                        text="Frame 1", 
                                        anchor="center",
                                        fg_color="transparent", 
                                        border_color="#3B8ED0", 
                                        border_width=2,  
                                        command=self.one_event)
        test1.grid(row=1, column=0, padx=(20, 20), pady=(15,15))

        test2 = customtkinter.CTkButton(self, 
                                        text="Frame 2", 
                                        anchor="center", 
                                        fg_color="transparent", 
                                        border_color="#3B8ED0", 
                                        border_width=2, 
                                        command=self.two_event)
        test2.grid(row=2, column=0, padx=(20, 20), pady=(15,15))

    def one_event(self):
        self.controller.frame1.tkraise()
         
    def two_event(self):
        self.controller.frame2.tkraise()

class Frame1(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="Frame 1!", 
                                       anchor="center")
        title.grid(row=0, column=0)
class Frame2(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        title = customtkinter.CTkLabel(self, 
                                       text="Frame 2!", 
                                       anchor="center")
        title.grid(row=0, column=0)

app = App()
app.mainloop()
