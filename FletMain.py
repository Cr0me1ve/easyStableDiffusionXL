import flet as ft
import torch
from diffusers import DiffusionPipeline
from scripts.runSD import genSingle
from screeninfo import get_monitors
import sys
import os
import cv2
import base64
import scripts.enteredData as ed

screenSize = [1920, 1080]
for m in get_monitors():
    if m.is_primary:
        screenSize[0] = m.width
        screenSize[1] = m.height
        break

pipe = None

def clearTemp():
    files = os.listdir(".temp")
    for file in files:
        os.remove(f".temp/{file}")
clearTemp()


def main(page: ft.Page):
    global pipe

    page.title = "Stable Diffusion"
    page.scroll = "auto"

    page.padding = 0
    page.window_width = screenSize[0] - screenSize[0]//3
    page.window_height = screenSize[1] - screenSize[1]//5
    page.window_resizable = True
    page.window_maximizable = True

    plug = ft.Text()

    ddSelectModel = ft.Dropdown(
        label="Model",
        width=300,
        options=[
            ft.dropdown.Option(name) for name in os.listdir("models/SD")
        ], scale=.8
    )

    pr = ft.ProgressRing(width=16, height=16, stroke_width = 2, value=0)
    modelLoadStatus = ft.Icon(name=ft.icons.CHECK_ROUNDED, color=ft.colors.GREEN, size=0)

    prompt = ft.TextField(label="Prompt", text_size=16)
    negPrompt = ft.TextField(label="Negative prompt", text_size=16)

    def updateImgs():
        images.controls.clear()
        genNum = 0
        for img in os.listdir(".temp"):
            if img.split(".")[-1] != "png":
                continue
            genNum += 1
            targetRes = 621
            imageCV2 = cv2.imread(f".temp/{img}")
            res = imageCV2.shape[0:2][::-1]     # [width, height]
            multiplyer = targetRes/max(res)
            renderRes = [int(res[0] * multiplyer), int(res[1] * multiplyer)]
            imgBase64 = base64.b64encode(cv2.imencode('.jpg', imageCV2)[1]).decode('utf-8')
            images.controls.append(
                ft.Column([
                    ft.Image(
                        src_base64=imgBase64,
                        width=renderRes[0],
                        height=renderRes[1],
                    ),
                    ft.Checkbox(width=renderRes[0], scale=1.3)
                ], alignment=ft.MainAxisAlignment.CENTER))
        if genNum > 0:
            saveBtn.scale = 1.3
        else:
            saveBtn.scale = 0.
        saveBtn.update()

    def btnLoadModel_clicked(e):
        global pipe
        pr.value = None
        page.update()
        try:
            pipe = loadPipe()
            loadModelBtn.disabled = True
            page.update()
        except:
            pr.value = 0
            modelLoadStatus.name = ft.icons.ERROR_ROUNDED
            modelLoadStatus.color = ft.colors.RED
            modelLoadStatus.size = 30
            page.update()

    loadModelBtn = ft.OutlinedButton(text="Load", on_click=btnLoadModel_clicked, scale=.8)

    def loadPipe():
        pipeline = DiffusionPipeline.from_pretrained("models/SD/" + ddSelectModel.value,
                                                     torch_dtype= torch.float16 if ddSelectModel.value.split("_")[-1] == "fp16" else torch.float32,
                                                     use_safetensors=True,
                                                     variant="fp16" if ddSelectModel.value.split("_")[-1] == "fp16" else "fp32")
        pipeline.enable_model_cpu_offload()
        pr.value = 0
        modelLoadStatus.name = ft.icons.CHECK_ROUNDED
        modelLoadStatus.color = ft.colors.GREEN
        modelLoadStatus.size = 30
        ddSelectModel.disabled = True
        return pipeline

    page.add(
        ft.Row([
            ddSelectModel,
            loadModelBtn,
            pr,
            modelLoadStatus
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.START, spacing=0)
    )

    page.add(
        prompt,
        negPrompt,
    )

    def widthChanged(e):
        widthText.value = f"Width: {int(widthSlider.value)}"
        widthText.update()
    
    def heightChanged(e):
        heightText.value = f"Height: {int(heightSlider.value)}"
        heightText.update()

    def sampleChanged(e):
        sampleText.value = f"Samples: {int(sampleSlider.value)}"
        sampleText.update()
    
    def countChanged(e):
        countText.value = f"Number of images: {int(countSlider.value)}"
        countText.update()

    widthSlider  = ft.Slider(min=512, max=2048, divisions=192, width=page.window_width//2, value=1280, on_change=widthChanged)
    heightSlider = ft.Slider(min=512, max=2048, divisions=192, width=page.window_width//2, value=720, on_change=heightChanged)
    sampleSlider = ft.Slider(min=10, max=120, divisions=22, width=page.window_width//2-16, value=50, on_change=sampleChanged)
    countSlider  = ft.Slider(min=1, max=20, divisions=19, width=page.window_width//2-16, value=1, on_change=countChanged)

    widthText  = ft.Text(f"Width: {widthSlider.value}", size=18, width=page.window_width//2, text_align=ft.TextAlign.CENTER)
    heightText = ft.Text(f"Height: {heightSlider.value}", size=18, width=page.window_width//2, text_align=ft.TextAlign.CENTER)
    sampleText = ft.Text(f"Samples: {sampleSlider.value}", size=18, width=page.window_width//2, text_align=ft.TextAlign.CENTER)
    countText = ft.Text(f"Number of images: {countSlider.value}", size=18, width=page.window_width//2, text_align=ft.TextAlign.CENTER)

    dlg = ft.AlertDialog()
    genPR = ft.ProgressRing(width=page.window_width//30, height=page.window_width//30, stroke_width = page.window_height//216, value=0)

    def genBtnClick(e):
        ed.saveToFile("enteredDataAutosave.json", prompt.value, negPrompt.value, widthSlider.value, heightSlider.value, sampleSlider.value, countSlider.value)
        clearTemp()
        updateImgs()
        if not loadModelBtn.disabled:
            dlg.title = ft.Text("You need to load the model first")
            page.dialog = dlg
            dlg.open = True
            page.update()
            return None
        
        genPR.value = None
        errorText.visible = False
        genBtn.disabled = True
        infoText.visible = True
        abortBtn.scale = 1.
        page.update()

        f = open('.temp/prompt.txt', 'w')
        f.write(f"{prompt.value}")
        f.close()

        try:
            for i in range(int(countSlider.value)):
                img = genSingle(pipe,
                                prompt=prompt.value,
                                negativePrompt=negPrompt.value,
                                width=int(widthSlider.value),
                                height=int(heightSlider.value),
                                samples=int(sampleSlider.value))
                img.save(f".temp/{i}.png")
                updateImgs()
                images.update()
        except Exception as ex:
            errorText.value = str(ex)
            errorText.visible = True
        
        abortBtn.scale = 0.
        genBtn.disabled = False
        genPR.value = 0
        infoText.visible = False
        updateImgs()
        page.update()

    def abortBtnClick(e):
        page.window_destroy()
        os.execv(sys.executable, ['python'] + [sys.argv[0].replace("\\", "/")])

    def saveBtnClick(e):
        imagesPrompt = open(".temp/prompt.txt").read()
        extraData = 0
        for i in range(len(images.controls)):
            if images.controls[i].controls[1].value:
                try:
                    img = cv2.imread(f'.temp/{i}.png')
                    while f"{imagesPrompt}_{i}{'' if extraData == 0 else '_' + str(extraData)}.png" in os.listdir("pics"):
                        extraData += 1
                    cv2.imwrite(f"pics/{imagesPrompt}_{i}{'' if extraData == 0 else '_' + str(extraData)}.png", img)
                except:
                    print(f"'{i}.png' does not exist")

    def saveSettingsBtnClick(e):
        ed.saveToFile("enteredDataSave.json", prompt.value, negPrompt.value, widthSlider.value, heightSlider.value, sampleSlider.value, countSlider.value)
        ed.saveToFile("enteredDataAutosave.json", prompt.value, negPrompt.value, widthSlider.value, heightSlider.value, sampleSlider.value, countSlider.value)

    def loadSettings(fileName: str):
        try:
            settings = ed.loadFromFile(fileName)
            prompt.value = settings["prompt"]
            negPrompt.value = settings["negPrompt"]
            widthSlider.value = settings["width"]
            heightSlider.value = settings["height"]
            sampleSlider.value = settings["samples"]
            countSlider.value = settings["count"]
            widthChanged("")
            heightChanged("")
            sampleChanged("")
            countChanged("")
            page.update()
        except Exception as e:
            print(e)

    def loadSettingBtnClick(e):
        loadSettings("enteredDataSave.json")

    def loadDefaultBtnClick(e):
        loadSettings("enteredDataDefault.json")

    genBtn = ft.ElevatedButton(text="Generate", height=50, scale=1.5, style=ft.ButtonStyle(
                color={
                    ft.MaterialState.HOVERED: ft.colors.BLACK,
                    ft.MaterialState.FOCUSED: ft.colors.WHITE,
                    ft.MaterialState.DEFAULT: ft.colors.BLACK,
                }, bgcolor=ft.colors.ORANGE), on_click=genBtnClick
            )

    abortBtn = ft.ElevatedButton(text="Abort", height=40, scale=0., style=ft.ButtonStyle(
                color={
                    ft.MaterialState.HOVERED: ft.colors.BLACK,
                    ft.MaterialState.FOCUSED: ft.colors.WHITE,
                    ft.MaterialState.DEFAULT: ft.colors.BLACK,
                }, bgcolor=ft.colors.RED_600), on_click=abortBtnClick
            )
    
    saveBtn = ft.ElevatedButton(text="Save selected", height=40, scale=0, style=ft.ButtonStyle(
                color={
                    ft.MaterialState.HOVERED: ft.colors.BLACK,
                    ft.MaterialState.FOCUSED: ft.colors.WHITE,
                    ft.MaterialState.DEFAULT: ft.colors.BLACK,
                }, bgcolor=ft.colors.GREEN_300), on_click=saveBtnClick
            )
    
    saveSettingsBtn = ft.ElevatedButton(text="Save", height=30, scale=1, on_click=saveSettingsBtnClick)
    loadSettingBtn = ft.ElevatedButton(text="Load", height=30, scale=1, on_click=loadSettingBtnClick)
    loadDefaultBtn = ft.ElevatedButton(text="Load default", height=30, scale=1, on_click=loadDefaultBtnClick)
    
    infoText = ft.Text(f"Generating. It may take some time.", size=16, visible=False)
    errorText = ft.Text(f"Generating. It may take some time.", size=16, visible=False)
    
    page.add(
        ft.Column([
            ft.Row([
                widthText,
                sampleText,
            ], spacing=0),
            ft.Row([
                widthSlider,
                sampleSlider,
            ], spacing=0, alignment=ft.MainAxisAlignment.START)
        ]),
        ft.Column([
            ft.Row([
                heightText,
                countText,
            ], spacing=0),
            ft.Row([
                heightSlider,
                countSlider,
            ], spacing=0, alignment=ft.MainAxisAlignment.START)
        ])
    )

    page.add(
        ft.Row([
            ft.Text(),
            saveSettingsBtn,
            loadSettingBtn,
            loadDefaultBtn
        ])
    )

    page.add(
        plug,
        ft.Column([
            ft.Row([
                plug,
                genBtn,
                genPR,
                abortBtn,
                saveBtn
            ], spacing=36),
            ft.Row([
                plug,
                infoText,
                errorText
            ], spacing=36),
        ])
    )

    images = ft.Row(alignment=ft.MainAxisAlignment.START, wrap=True)
    updateImgs()
    page.add(images)

    loadSettings("enteredDataAutosave.json")

ft.app(target=main)