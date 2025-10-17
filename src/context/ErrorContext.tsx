import { createContext, useContext, useState, type ReactNode } from "react";

type ErrorContextType = {
    error: ErrorType;
    addError: (errorMessage: string, code?: ErrorCode) => void;
    clearError: () => void;
};

const ErrorContext = createContext<ErrorContextType | undefined>(undefined);

type Props = {
    children: ReactNode
}

type ErrorCode = 'UNAUTHORIZED' | 'SERVER_ERROR' | 'NETWORK_ERROR' | 'NOT_FOUND' | 'UNKNOWN';

type ErrorType = {
    message: string;
    timestamp: string;
    code?: ErrorCode;
} | null;

export const ErrorProvider = ({ children }: Props) => {
    const [error, setError] = useState<ErrorType>(null);


    const addError = (errorMessage: string, code?: ErrorCode) => {
        setError({ message: errorMessage, code, timestamp: new Date().toISOString() });
    };

    const clearError = () => {
        setError(null);
    };

    return (
        <ErrorContext.Provider value={{ error, addError, clearError }}>
            {children}
        </ErrorContext.Provider>
    );
};

export const useError = () => {
    const context = useContext(ErrorContext);
    if (!context) {
        throw new Error("useError must be used within an ErrorProvider");
    }
    return context;
};
