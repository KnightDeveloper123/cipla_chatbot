import { type ReactNode } from "react";
import { ErrorProvider } from "@/context";

const AppProviders = ({ children }: { children: ReactNode }) => {
    return (
        <ErrorProvider>
            {children}
        </ErrorProvider>
    );
};

export default AppProviders;
