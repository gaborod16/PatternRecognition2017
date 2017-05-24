function varargout = RP_gui(varargin)
% RP_GUI MATLAB code for RP_gui.fig
%      RP_GUI, by itself, creates a new RP_GUI or raises the existing
%      singleton*.
%
%      H = RP_GUI returns the handle to a new RP_GUI or the handle to
%      the existing singleton*.
%
%      RP_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RP_GUI.M with the given input arguments.
%
%      RP_GUI('Property','Value',...) creates a new RP_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RP_gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RP_gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RP_gui

% Last Modified by GUIDE v2.5 21-May-2017 10:48:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RP_gui_OpeningFcn, ...
                   'gui_OutputFcn',  @RP_gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before RP_gui is made visible.
function RP_gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RP_gui (see VARARGIN)

% Choose default command line output for RP_gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes RP_gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = RP_gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in run.
function run_Callback(hObject, eventdata, handles)
% hObject    handle to run (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
panel1 = get(handles.scenario, 'SelectedObject');
scene = get(panel1,'Tag')

panel2 = get(handles.featureSelection, 'SelectedObject');
fselect = get(panel2,'Tag')
nfeatures= str2num(get(handles.nFeatures,'String'))

panel3 = get(handles.classifiers, 'SelectedObject');
classifier = get(panel3,'Tag')

load read_source.mat;
[data, meta] = FeatureProcess.RemCorrelated(data,meta);

% Scenario
if scene=="binary"
    binary=1;
else
    binary=0;
end

%Features Selection
if fselect=="pca"
    features = FeatureProcess.PCA(data,nfeatures,binary);
    
elseif fselect=="kruskal"
    features = FeatureProcess.KruskalWallis(data,meta,nfeatures,binary);

elseif fselect=="lda"
    features = FeatureProcess.LDA(data,nfeatures,binary);
end
        
%Classification
if classifier=="euclidean"
    
    [test_result,conf_matrix, error]=Classifier.MinDistEuc(features,1);
    
elseif classifier=="mahalanobis"
    [test_result,conf_matrix, error]=Classifier.MinDistMah(features,1);
    
elseif classifier=="fisher"
    [test_result,conf_matrix, error]=Classifier.FisherLD(features,1);
    
elseif classifier=="svm"
    [test_result,conf_matrix, error]=Classifier.SupportVM(features,1);
    
elseif classifier=="bayes"
    [test_result,conf_matrix, error]=Classifier.Bayesian(features,1);
    
elseif classifier=="knn"
    [test_result,conf_matrix, error]=Classifier.KNearestNeighboors(features,1);
    
elseif classifier=="hybrid"
    [test_result,conf_matrix, error]=Classifier.HybridClassifier(features,1);

elseif classifier=="divide"
    [test_result, conf_matrix, error] = Classifier.DivideConquer(data, nfeatures, 'Classifier.Bayesian')
end

accuracy=1-error;
texto= sprintf(horzcat( '\n \n \n \n Error: ', mat2str(round(error,3)), '\n \n Accuracy: ',mat2str(round(accuracy,3))));

set(handles.results,'String',texto);



% --- Executes during object creation, after setting all properties.
function results_CreateFcn(hObject, eventdata, handles)
% hObject    handle to results (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called



function nFeatures_Callback(hObject, eventdata, handles)
% hObject    handle to nFeatures (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of nFeatures as text
%        str2double(get(hObject,'String')) returns contents of nFeatures as a double


% --- Executes during object creation, after setting all properties.
function nFeatures_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nFeatures (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
