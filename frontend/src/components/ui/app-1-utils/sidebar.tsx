import {createContext,useContext,useState,type PropsWithChildren,type HTMLAttributes} from "react";
import {PanelLeft} from "lucide-react";
const Side=createContext({open:false,toggle:()=>{}});
export const useSidebar=()=>useContext(Side);
export function SidebarProvider({children}:PropsWithChildren){const [open,set]=useState(()=>window.innerWidth>=768);return <Side.Provider value={{open,toggle:()=>set(v=>!v)}}><div className="flex min-h-screen">{children}</div></Side.Provider>}
export function SidebarInset(props:HTMLAttributes<HTMLDivElement>){return <div {...props} className={"flex-1 "+props.className}/>}
export function SidebarTrigger(props:HTMLAttributes<HTMLButtonElement>){const {open,toggle}=useSidebar();return <button {...props} className="p-2" aria-expanded={open} aria-controls="navigation" onClick={toggle}><PanelLeft size={20}/></button>}
